#ifndef LLM_H
#define LLM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/* ============================================================================
 * 1. Config: 트랜스포머 모델의 크기(하이퍼파라미터) 설정 구조체
 * ============================================================================ */
typedef struct {
	int vocab_size;   // 단어장 크기 (모델이 아는 전체 고유 단어/토큰 개수)
	int dim;          // 모델의 기본 임베딩 벡터 차원 (d_model)
	int hidden_dim;   // FFN(피드포워드 네트워크) 내부의 확장 차원
	int seq_len;      // 한 번에 입력받는 문장의 토큰 길이 (행 개수, row)
	int window_size;  // 슬라이딩 윈도우 어텐션에서 참고할 최대 과거 토큰 범위
} Config;

/* ============================================================================
 * 2. Weights: 모델이 학습을 통해 업데이트하는 모든 가중치(파라미터) 구조체
 * ============================================================================ */
typedef struct {
	// [입출력 가중치]
	double* token_embedding; // [vocab_size * dim] : 각 단어 번호(ID)에 대응하는 임베딩 벡터 표
	double* final_gamma;     // [dim]              : 최종 출력 직전 RMSNorm의 스케일(배율) 가중치
	double* W_out;           // [dim * vocab_size] : 최종 벡터를 단어장 점수로 변환하는 출력 가중치

	// [어텐션(Attention) 블록 가중치]
	double* Wq;              // [dim * dim]        : 입력에서 Query(질문)를 만드는 가중치 행렬
	double* Wk;              // [dim * dim]        : 입력에서 Key(꼬리표)를 만드는 가중치 행렬
	double* Wv;              // [dim * dim]        : 입력에서 Value(실제 의미 값)를 만드는 가중치 행렬
	double* Wo;              // [dim * dim]        : 어텐션 결과를 최종 투영(Projection)하는 가중치 행렬

	// [FFN(GeGLU 피드포워드) 블록 가중치]
	double* W_gate;          // [dim * hidden_dim] : GELU 문지기(Gate) 값을 만드는 가중치 행렬
	double* W_up;            // [dim * hidden_dim] : 차원을 hidden_dim으로 확장하는 가중치 행렬
	double* W_down;          // [hidden_dim * dim] : 확장된 차원을 다시 dim으로 압축하는 가중치 행렬

	// [서브 블록 앞뒤 RMSNorm 스케일 가중치 (Gemma 2 Pre/Post-Norm 구조)]
	double* gamma_att_in;    // [dim]              : 어텐션 블록 입력 전 RMSNorm 배율
	double* gamma_att_out;   // [dim]              : 어텐션 블록 출력 후 RMSNorm 배율
	double* gamma_ffn_in;    // [dim]              : FFN 블록 입력 전 RMSNorm 배율
	double* gamma_ffn_out;   // [dim]              : FFN 블록 출력 후 RMSNorm 배율
} Weights;

/* ============================================================================
 * 3. Activations: 순전파(Forward) 도중 계산되는 중간 결과값 저장 버퍼 구조체
 *    (역전파 때 미분 계산의 재료로 재사용하기 위해 보존함)
 * ============================================================================ */
typedef struct {
	double* X;               // [seq_len * dim]        : 임베딩 조회 및 sqrt(dim) 스케일링이 끝난 입력 행렬

	// [어텐션 블록 중간 결과물]
	double* norm_att_in;     // [seq_len * dim]        : 어텐션 입력 정규화(RMSNorm) 결과
	double* Q;               // [seq_len * dim]        : Query 행렬 (norm_att_in * Wq)
	double* K;               // [seq_len * dim]        : Key 행렬 (norm_att_in * Wk)
	double* V;               // [seq_len * dim]        : Value 행렬 (norm_att_in * Wv)
	double* Q_rope;          // [seq_len * dim]        : RoPE(회전 위치 인코딩)가 적용된 Query
	double* K_rope;          // [seq_len * dim]        : RoPE(회전 위치 인코딩)가 적용된 Key
	double* K_T;             // [dim * seq_len]        : K_rope를 전치(Transpose)한 행렬
	double* scores;          // [seq_len * seq_len]    : Q_rope * K_T 후 1/sqrt(dim)로 스케일링된 어텐션 점수
	double* capped_scores;   // [seq_len * seq_len]    : Softcap(50.0)으로 상한선이 제한된 어텐션 점수
	double* masked_scores;   // [seq_len * seq_len]    : 미래 토큰 및 윈도우 밖을 -1e9로 가린(Causal Mask) 점수
	double* probs;           // [seq_len * seq_len]    : Softmax를 거쳐 확률(0~1)로 변환된 어텐션 가중치
	double* attn_out;        // [seq_len * dim]        : 확률(probs)과 Value(V)를 곱해 결합한 문맥 벡터
	double* proj_out;        // [seq_len * dim]        : 어텐션 출력 가중치(Wo)를 곱한 결과
	double* norm_att_out;    // [seq_len * dim]        : 어텐션 출력 후 정규화(RMSNorm) 결과
	double* res_att;         // [seq_len * dim]        : 첫 번째 잔차 연결(X + norm_att_out) 결과

	// [FFN 블록 중간 결과물]
	double* norm_ffn_in;     // [seq_len * dim]        : FFN 입력 정규화(RMSNorm) 결과
	double* gate;            // [seq_len * hidden_dim] : Gate 선형 변환 결과 (norm_ffn_in * W_gate)
	double* gate_act;        // [seq_len * hidden_dim] : Gate에 GELU 활성화 함수를 통과시킨 결과
	double* up;              // [seq_len * hidden_dim] : Up 선형 변환 결과 (norm_ffn_in * W_up)
	double* hidden;          // [seq_len * hidden_dim] : gate_act와 up을 원소별 곱셈(Hadamard)한 은닉 상태
	double* ffn_out;         // [seq_len * dim]        : Down 선형 변환으로 차원을 복구한 결과 (hidden * W_down)
	double* norm_ffn_out;    // [seq_len * dim]        : FFN 출력 후 정규화(RMSNorm) 결과
	double* res_ffn;         // [seq_len * dim]        : 두 번째 잔차 연결(res_att + norm_ffn_out) 결과

	// [최종 출력층 중간 결과물]
	double* final_norm;      // [seq_len * dim]        : 최종 RMSNorm을 통과한 결과
	double* raw_logits;      // [seq_len * vocab_size] : 단어장 크기만큼 변환된 날것의 점수 (final_norm * W_out)
	double* logits;          // [seq_len * vocab_size] : 최종 Softcap(30.0)을 거친 로짓 점수
	double* out_probs;       // [seq_len * vocab_size] : 최종 Softmax를 거친 각 단어별 예측 확률 분포
} Activations;

/* ============================================================================
 * 4. Gradients: 역전파(Backward) 과정에서 계산되는 모든 기울기(미분값) 저장 구조체
 * ============================================================================ */
typedef struct {
	// [가중치(Weights)에 대한 기울기 — update_weights에서 가중치 수정에 사용됨]
	double* d_token_embedding; // token_embedding의 기울기
	double* d_final_gamma;     // final_gamma의 기울기
	double* d_W_out;           // W_out의 기울기
	
	double* d_Wq;              // Wq의 기울기
	double* d_Wk;              // Wk의 기울기
	double* d_Wv;              // Wv의 기울기
	double* d_Wo;              // Wo의 기울기

	double* d_W_gate;          // W_gate의 기울기
	double* d_W_up;            // W_up의 기울기
	double* d_W_down;          // W_down의 기울기

	double* d_gamma_att_in;    // gamma_att_in의 기울기
	double* d_gamma_att_out;   // gamma_att_out의 기울기
	double* d_gamma_ffn_in;    // gamma_ffn_in의 기울기
	double* d_gamma_ffn_out;   // gamma_ffn_out의 기울기

	// [최종 출력층 역전파용 중간 기울기]
	double* d_logits;          // logits에 대한 기울기 (Cross-Entropy + Softmax 미분 결과)
	double* d_raw_logits;      // raw_logits에 대한 기울기
	double* d_final_norm;      // final_norm에 대한 기울기
	double* d_res_ffn;         // res_ffn에 대한 기울기

	// [FFN 블록 역전파용 중간 기울기]
	double* d_norm_ffn_out;    // norm_ffn_out에 대한 기울기
	double* d_ffn_out;         // ffn_out에 대한 기울기
	double* d_hidden;          // hidden에 대한 기울기
	double* d_gate_act;        // gate_act에 대한 기울기
	double* d_gate;            // gate에 대한 기울기
	double* d_up;              // up에 대한 기울기
	double* d_norm_ffn_in;     // norm_ffn_in에 대한 기울기
	double* d_res_att;         // res_att에 대한 기울기
	
	// [어텐션 블록 역전파용 중간 기울기]
	double* d_norm_att_out;    // norm_att_out에 대한 기울기
	double* d_proj_out;        // proj_out에 대한 기울기
	double* d_attn_out;        // attn_out에 대한 기울기
	double* d_probs;           // probs에 대한 기울기
	double* d_scores;          // scores( 및 capped/masked scores)에 대한 기울기
	double* d_Q_rope;          // Q_rope에 대한 기울기
	double* d_K_rope;          // K_rope에 대한 기울기
	double* d_Q;               // Q에 대한 기울기
	double* d_K;               // K에 대한 기울기
	double* d_V;               // V에 대한 기울기
	double* d_norm_att_in;     // norm_att_in에 대한 기울기
	double* d_X;               // 입력 행렬 X에 대한 기울기

	// [역전파 중 행렬 전치(Transpose) 및 누적 계산에 사용하는 공용 임시 버퍼]
	double* temp1;
	double* temp2;
} Gradients;

/* ============================================================================
 * 5. 함수 선언 (Function Prototypes)
 * ============================================================================ */

// [기본 행렬 및 수학 연산 함수]
int transpose(double* input, double* output, int row, int col);
int softmax(double* input, double* output, int count);
int matmul(double* input1, double* input2, double* output, int row, int dk, int col);
int matadd(double* input1, double* input2, double* output, int row, int col);
int matsub(double* input1, double* input2, double* output, int row, int col);
int hadamard(double* input1, double* input2, double* output, int row, int col);
int scalemul(double* input, double* output, double scale, int row, int col);
int scaleadd(double* input, double* output, double scale, int row, int col);
int matdiv(double* input1, double* input2, double* output, int row, int col);
int mean(double* input, double* output, int row, int col);
int dot(double* input1, double* input2, double* result, int count);
int argmax(double* input, int count);

// [트랜스포머 핵심 레이어 순전파(Forward) 연산 함수]
int rmsnorm(double* input, double* gamma, double* output, int row, int col);
int gelu(double* input, double* output, int row, int col);
int softcap(double* input, double* output, double cap, int row, int col);
int rope(double* input, double* output, int row, int col);
int causal_mask(double* input, double* output, int row, int col, int window_size);

// [트랜스포머 핵심 레이어 역전파(Backward) 미분 함수]
int cross_entropy_backward(double* probs, double* d_in, int target, int count);
int softcap_backward(double* input, double* d_out, double* d_in, double cap, int row, int col);
int rope_backward(double* d_out, double* d_in, int row, int col);
int softmax_backward(double* probs, double* d_out, double* d_in, int count);
int gelu_backward(double* input, double* d_out, double* d_in, int row, int col);
int rmsnorm_backward(double* input, double* gamma, double* d_out, double* d_in, double* d_gamma, int row, int col);

// [가중치 초기화 및 경사하강법(SGD) 보조 함수]
int init_weight(double* weight, int row, int col);
int init_gamma(double* gamma, int col);
int zero_grad(double* grad, int row, int col);
int sgd_update(double* weight, double* grad, double lr, int row, int col);

// [블록 단위 및 전체 모델 순전파(Forward) 조립 함수]
int attention(Config* cfg, Weights* w, Activations* act);
int ffn(Config* cfg, Weights* w, Activations* act);
int forward(int* tokens, Config* cfg, Weights* w, Activations* act);

// [블록 단위 및 전체 모델 역전파(Backward) 조립 함수]
int ffn_backward(Config* cfg, Weights* w, Activations* act, Gradients* grad);
int attention_backward(Config* cfg, Weights* w, Activations* act, Gradients* grad);
int backward(int* tokens, int* targets, Config* cfg, Weights* w, Activations* act, Gradients* grad);

// [모델 전체 가중치 업데이트 및 메모리 할당/초기화 함수]
int update_weights(Config* cfg, Weights* w, Gradients* grad, double lr);
int init_model(Config* cfg, Weights* w, Activations* act, Gradients* grad);

#endif


