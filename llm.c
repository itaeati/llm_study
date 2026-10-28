#include "llm.h"

/* ============================================================================
 * [1] 기본 행렬 및 벡터 연산 함수 (12개)
 * ============================================================================ */

// 행렬 전치 (Transpose): [row * col] 크기의 입력 행렬을 행과 열을 뒤집어 [col * row]로 저장
int transpose(double* input, double* output, int row, int col)
{
	if (input == NULL || output == NULL || row <= 0 || col <= 0)
		return -1;

	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			output[j * row + i] = input[i * col + j];
	return 0;
}

// 소프트맥스 (Softmax): 1차원 배열(count 길이)의 점수들을 합이 1.0이 되는 확률 분포(0.0~1.0)로 변환
// (지수 함수 오버플로우 방지를 위해 최댓값 max_val을 먼저 빼주는 안정화 기법 적용)
int softmax(double* input, double* output, int count)
{
	if (input == NULL || output == NULL || count <= 0)
		return -1;

	double temp = 0, max_val = input[0];

	for (int i = 1; i < count; i++)
	{
		if (input[i] > max_val)
			max_val = input[i];
	}

	for (int j = 0; j < count; j++)
	{
		output[j] = exp(input[j] - max_val);
		temp += output[j];
	}
	
	for (int i = 0; i < count; i++)
		output[i] /= temp;

	return 0;
}

// 행렬 곱셈 (Matrix Multiplication): [row * dk] 행렬(input1)과 [dk * col] 행렬(input2)을 곱해 [row * col] 행렬(output) 생성
int matmul(double* input1, double* input2, double* output, int row, int dk, int col)
{
	if (input1 == NULL || input2 == NULL || output == NULL || row <= 0 || dk <= 0 || col <= 0)
		return -1;

	for (int i  = 0; i < row; i++)
	{
		for (int j  = 0; j < col; j++)
		{
			double temp = 0;
			for (int k = 0; k < dk; k++)
				temp += input1[i * dk + k] * input2[k * col + j];

			output[i * col + j] = temp;
		}
	}
	return 0;
}

// 행렬 덧셈: 같은 크기 [row * col]의 두 행렬을 같은 위치끼리 더함 (잔차 연결 및 기울기 누적에 사용)
int matadd (double* input1, double* input2, double* output, int row, int col)
{
	if (input1 == NULL || input2 == NULL || output == NULL || row <= 0 || col <= 0)
		return -1;

	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			output[i * col + j] = input1[i * col + j] + input2[i * col + j];
	return 0;
}

// 행렬 뺄셈: 같은 크기 [row * col]의 두 행렬을 같은 위치끼리 뺌
int matsub (double* input1, double* input2, double* output, int row, int col)
{
	if (input1 == NULL || input2 == NULL || output == NULL || row <= 0 || col <= 0)
		return -1;

	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			output[i * col + j] = input1[i * col + j] - input2[i * col + j];
	return 0;
}

// 아다마르 곱 (Hadamard Product): 같은 크기 [row * col]의 두 행렬을 같은 위치의 원소끼리 곱함 (GeGLU 문지기 연산 등에 사용)
int hadamard(double* input1, double* input2, double* output, int row, int col)
{
	if (input1 == NULL || input2 == NULL || output == NULL || row <= 0 || col <= 0)
		return -1;

	for (int i  = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			output[i * col + j] = input1[i * col + j] * input2[i * col + j];
	return 0;
}

// 스칼라 곱셈: 행렬의 모든 원소에 상수(scale)를 곱함 (임베딩 스케일링 및 어텐션 점수 1/sqrt(dim) 스케일링에 사용)
int scalemul(double* input, double* output, double scale, int row, int col)
{
	if (input == NULL || output == NULL || row <= 0 || col <= 0)
		return -1;

	for (int i  = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			output[i * col + j] = input[i * col + j] * scale;
	return 0;
}

// 스칼라 덧셈: 행렬의 모든 원소에 상수(scale)를 더함
int scaleadd (double* input, double* output, double scale, int row, int col)
{
	if (input == NULL || output == NULL || row <= 0 || col <= 0)
		return -1;

	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			output[i * col + j] = input[i * col + j] + scale;
	return 0;
}

// 원소별 나눗셈: input1의 각 원소를 input2의 같은 위치 원소로 나눔 (0으로 나누기 방지 포함)
int matdiv (double* input1, double* input2,  double* output, int row, int col)
{
	if (input1 == NULL || input2 == NULL || output == NULL || row <= 0 || col <= 0)
		return -1;

	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
		{
			if (input2[i * col + j] == 0.0)
				return -1;
			output[i * col + j] = (input1[i * col + j] / input2[i * col + j]);
		}
	return 0;
}

// 행별 평균 (Mean): 각 행(row)마다 col개 원소의 평균을 구해 output[row]에 저장
int mean (double* input, double* output, int row, int col)
{
	if (input == NULL || output == NULL || row <= 0 || col <= 0)
		return -1;

	for (int i = 0; i < row; i++)
	{
		double temp = 0;
		for (int j = 0; j < col; j++)
			temp += input[i * col + j];

		output[i] = (1.0 / col) * temp;
	}
	return 0;
}

// 벡터 내적 (Dot Product): 길이 count인 두 1차원 벡터의 내적값을 구해 *result에 저장
int dot (double* input1, double* input2, double * result, int count)
{
	if (input1 == NULL || input2 == NULL || result == NULL || count <= 0)
		return -1;

	double temp = 0;
	for (int i = 0; i < count; i++)
		temp += input1[i] * input2[i];
	*result = temp;

	return 0;
}

// 최댓값 인덱스 찾기 (Argmax): 배열에서 가장 큰 값(가장 높은 확률)이 들어있는 인덱스 번호를 반환
int argmax (double* input, int count)
{
	if (input == NULL || count <= 0)
		return -1;

	int index = 0;
	for (int i = 1; i < count; i++)
		if (input[i] > input[index])
			index = i;
	return index;
}

/* ============================================================================
 * [2] 트랜스포머 핵심 레이어 순전파(Forward) 연산 함수 (5개)
 * ============================================================================ */

// RMSNorm 정규화: 각 행 벡터의 제곱평균제곱근(RMS)으로 나누어 크기를 일정하게 맞추고 학습 가중치(gamma)를 곱함
int rmsnorm (double* input, double* gamma, double* output, int row, int col)
{
	if (input == NULL || gamma == NULL || output == NULL || row <= 0 || col <= 0)
		return -1;

	for (int i  = 0; i < row; i++)
	{
		double temp = 0, rms = 0;
		for (int j = 0; j < col; j++)
			temp += input[i * col + j] * input[i * col + j];

		rms = sqrt(((1.0 / col) * temp) + 1e-6);
			
		for (int j = 0; j < col; j++)
			output[i * col + j] = (input[i * col + j] / rms) * gamma[j];
		
	}
	return 0;
}

// GELU 활성화 함수 (Tanh 근사식): FFN의 Gate에서 음수는 부드럽게 억제하고 양수는 통과시키는 비선형 활성화 함수
int gelu(double* input, double* output, int row, int col)
{
	if (input == NULL || output == NULL || row <= 0 || col <= 0)
		return -1;

	for (int i  = 0; i < row; i++)
		for (int j  = 0; j < col; j++)
			output[i * col + j] = 0.5 * input[i * col + j] * (1.0 + tanh(0.79788456 * (input[i * col + j] + 0.044715 * (input[i * col + j] * input[i * col + j] * input[i * col + j]))));
	return 0;
}

// Softcap (Gemma 2 고유 기법): 점수(input)가 지나치게 커지지 않도록 cap * tanh(x / cap)으로 부드럽게 상한선을 제한
int softcap(double* input, double* output, double cap, int row, int col)
{
	if (input == NULL || output == NULL || cap <= 0.0 || row <= 0 || col <= 0)
		return -1;

	for (int i  = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			output[i * col + j] = cap * tanh(input[i * col + j] / cap);
	return 0;
}

// RoPE (Rotary Position Embedding): 토큰의 위치(i)에 따라 벡터의 짝수·홀수 차원 쌍을 회전시켜 상대적 위치 정보를 주입
int rope(double* input, double* output, int row, int col)
{
	if (input == NULL || output == NULL || row <= 0 || col <= 0 || col % 2 != 0)
		return -1;

	double seta = 0, phi = 0;

	for (int i = 0; i < row; i++)
		for (int j = 0; j < col/2; j++)
		{
			seta = 1.0 / pow(10000.0, (2.0 * j)/col);
			phi = i * seta;
			output[i * col + 2 * j] = input[i * col + 2 * j] * cos(phi) - input[i * col + 2 * j + 1] * sin(phi);
			output[i * col + 2 * j + 1] = input[i * col + 2 * j + 1] * cos(phi) + input[i * col + 2 * j] * sin(phi);
		}
	return 0;
}

// Causal + Sliding Window Mask: 미래의 단어(j > i)나 윈도우 범위 밖의 오래된 단어(i - j >= window_size)를
// 볼 수 없도록 점수를 -1e9(Softmax 통과 시 확률 0)로 가려버림
int causal_mask(double* input, double* output, int row, int col, int window_size)
{
	if (input == NULL || output == NULL || row <= 0 || col <= 0 || window_size <= 0)
		return -1;

	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
		{
			if (j > i || i - j >= window_size)
				output[i * col + j] = -1e9;
			else 
				output[i * col + j] = input[i * col + j];
		}
	return 0;
}

/* ============================================================================
 * [3] 트랜스포머 핵심 레이어 역전파(Backward) 미분 함수 (6개)
 * ============================================================================ */

// Cross-Entropy + Softmax 통합 역전파: 예측 확률(probs)에서 정답 위치(target)만 1.0을 빼서(p - y) 초기 오차 기울기를 구함
int cross_entropy_backward(double* probs, double* d_in, int target, int count)
{
	if (probs == NULL || d_in == NULL || count <= 0 || target < 0 || target >= count)
		return -1;

	for (int i = 0; i < count; i++)
	{
		if (i == target)
			d_in[i] = probs[i] - 1.0;
		else 
			d_in[i] = probs[i];
	}
	return 0;
}

// Softcap 역전파: y = cap * tanh(x / cap)의 미분인 dy/dx = 1 - tanh^2(x / cap)을 상류 기울기(d_out)에 곱함
int softcap_backward(double* input, double* d_out, double* d_in, double cap, int row, int col)
{
	if (input == NULL || d_out == NULL || d_in == NULL || cap <= 0.0 || row <= 0 || col <= 0)
		return -1;

	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			d_in[i * col + j] = d_out[i * col + j] * (1.0 - (tanh(input[i * col + j] / cap) * tanh(input[i * col + j] / cap)));
	return 0;
}

// RoPE 역전파: 순전파 때 +phi만큼 회전시켰던 기울기 벡터를 반대 방향(-phi)으로 역회전시켜 원래 좌표계로 되돌림
int rope_backward(double* d_out, double* d_in, int row, int col)
{
	if (d_out == NULL || d_in == NULL || row <= 0 || col <= 0 || col % 2 != 0)
		return -1;

	for (int i = 0; i < row; i++)
	{
		double seta = 0, phi = 0;
		for (int j = 0; j < col / 2; j++)
		{
			seta = 1.0 / pow(10000.0, (2.0 * j)/col);
			phi = i * seta;
			d_in[i * col + 2 * j] = d_out[i * col + 2 * j] * cos(phi) + d_out[i * col + 2 * j + 1] * sin(phi);
			d_in[i * col + 2 * j + 1] = d_out[i * col + 2 * j + 1] * cos(phi) - d_out[i * col + 2 * j] * sin(phi);
		}
	}
	return 0;
}

// Softmax 역전파 (어텐션 확률 미분용): d_in[i] = probs[i] * (d_out[i] - sum(probs * d_out))
int softmax_backward(double* probs, double* d_out, double* d_in, int count)
{
	if (probs == NULL || d_out == NULL || d_in == NULL || count <= 0)
		return -1;

	double temp = 0;
	for (int i = 0; i < count; i++)
		temp += probs[i] * d_out[i];
	for (int i = 0; i < count; i++)
		d_in[i] = probs[i] * (d_out[i] - temp);
	return 0;
}

// GELU 역전파: Tanh 근사 GELU 수식의 도함수를 계산하여 상류 기울기(d_out)와 곱함
int gelu_backward(double* input, double* d_out, double* d_in, int row, int col)
{
	if (input == NULL || d_out == NULL || d_in == NULL || row <= 0 || col <= 0)
		return -1;

	double t = 0, dt = 0;
	for (int i  = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			t = tanh(0.79788456 * (input[i * col + j] + 0.044715 * input[i * col + j] * input[i * col + j] * input[i * col + j]));
			dt = 0.5 * input[i * col + j] * (1.0 - t * t) * 0.79788456 * (1.0 + 3.0 * 0.044715 * input[i * col + j] * input[i * col + j]);
			d_in[i * col + j] = d_out[i * col + j] * (0.5 * (1.0 + t) + dt);
		}
	}
	return 0;
}

// RMSNorm 역전파: 스케일 가중치의 기울기(d_gamma)를 누적하고, 입력에 대한 기울기(d_in)를 계산
int rmsnorm_backward(double* input, double* gamma, double* d_out, double* d_in, double* d_gamma, int row, int col)
{
	if (input == NULL || gamma == NULL || d_out == NULL || d_in == NULL || d_gamma == NULL || row <= 0 || col <= 0)
		return -1;

	for (int i = 0; i < row; i++)
	{
		double temp = 0, rms = 0, sum_val = 0;
		for (int j = 0; j < col; j++)
			temp += input[i * col + j] * input[i * col + j];
		rms = sqrt(((1.0 / col) * temp) + 1e-6);

		for (int j = 0; j < col; j++)
		{
			d_gamma[j] += d_out[i * col + j] * (input[i * col + j] / rms);
			sum_val += d_out[i * col + j ] * gamma[j] * input[i * col + j];
		}

		for (int j = 0; j < col; j++)
			d_in[i * col + j] = (1.0 / rms) * (d_out[i* col + j] * gamma[j] - (input[i * col + j] * sum_val)/(col*rms*rms));
	}
	return 0;
}

/* ============================================================================
 * [4] 가중치 초기화 및 경사하강법(SGD) 보조 함수 (4개)
 * ============================================================================ */

// 행렬 가중치 난수 초기화 (Xavier/Glorot 유사 스케일): [-1/sqrt(col), +1/sqrt(col)] 범위의 균일 난수로 초기화
int init_weight(double* weight, int row, int col)
{
	if (weight == NULL || row <= 0 || col <= 0)
		return -1;

	double scale = 1.0 / sqrt(col);

	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			weight[i * col + j] = (((double)rand()/RAND_MAX)* 2.0 - 1.0) * scale;

	return 0;
}

// RMSNorm 배율 가중치(gamma) 초기화: 처음에는 원래 크기를 유지하도록 모두 1.0으로 초기화
int init_gamma(double* gamma, int col)
{
	if (gamma == NULL || col <= 0)
		return -1;

	for (int i = 0; i < col; i++)
		gamma[i] = 1.0;
	return 0;
}

// 기울기 버퍼 0.0 초기화: 역전파 때(+=) 이전 에폭의 기울기가 섞이지 않도록 0.0으로 비움
int zero_grad(double* grad, int row, int col)
{
	if (grad == NULL || row <= 0 || col <= 0)
		return -1;

	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			grad[i * col + j] = 0.0;
	return 0;
}

// 확률적 경사하강법(SGD) 가중치 업데이트: weight = weight - (학습률 lr * 기울기 grad)
int sgd_update (double* weight, double* grad, double lr, int row, int col)
{
	if (weight == NULL || grad == NULL || lr <= 0.0 || row <= 0 || col <= 0)
		return -1;

	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			weight[i * col + j] -= lr * grad[i * col + j];
	return 0;
}

/* ============================================================================
 * [5] 트랜스포머 블록 및 전체 모델 순전파(Forward) 조립 함수 (3개)
 * ============================================================================ */

// 어텐션(Attention) 서브 블록 순전파: 입력(X) -> Pre-RMSNorm -> Q,K,V -> RoPE -> Scaled Dot-Product
// -> Softcap(50.0) -> Causal Mask -> Softmax -> V 결합 -> Wo 투영 -> Post-RMSNorm -> 잔차 연결(res_att)
int attention(Config* cfg, Weights* w, Activations* act)
{
	if (cfg == NULL || w == NULL || act == NULL)
		return -1;
	if (cfg->seq_len <= 0 || cfg->dim <= 0 || cfg->window_size <= 0)
		return -1;

	int row = cfg->seq_len;
	int dim = cfg->dim;

	// 1. 어텐션 입력 전 정규화 (Pre-Attention RMSNorm)
	if (rmsnorm(act->X, w->gamma_att_in, act->norm_att_in, row, dim) < 0)
		return -1;

	// 2. Query, Key, Value 행렬 생성
	if (matmul(act->norm_att_in, w->Wq, act->Q, row, dim, dim) < 0)
		return -1;
	if (matmul(act->norm_att_in, w->Wk, act->K, row, dim, dim) < 0)
		return -1;
	if (matmul(act->norm_att_in, w->Wv, act->V, row, dim, dim) < 0)
		return -1;

	// 3. Query와 Key에 회전 위치 인코딩(RoPE) 적용
	if (rope(act->Q, act->Q_rope, row, dim) < 0)
		return -1;
	if (rope(act->K, act->K_rope, row, dim) < 0)
		return -1;

	// 4. Key 전치 후 Q * K^T 행렬곱 및 1/sqrt(dim) 스케일링으로 어텐션 점수(scores) 계산
	if (transpose(act->K_rope, act->K_T, row, dim)< 0)
		return -1;
	if (matmul(act->Q_rope, act->K_T, act->scores, row, dim, row) < 0)
		return -1;
	if (scalemul(act->scores, act->scores, 1.0 / sqrt(dim), row, row) < 0)
		return -1;

	// 5. 어텐션 점수 Softcap(50.0) 및 미래 토큰 마스킹(Causal Mask)
	if (softcap(act->scores, act->capped_scores, 50.0, row, row) < 0)
		return -1;
	if (causal_mask(act->capped_scores, act->masked_scores, row, row, cfg->window_size) < 0)
		return -1;

	// 6. 각 행(토큰)별로 Softmax를 취해 어텐션 확률(probs) 계산
	for (int i = 0; i < row; i++)
		if (softmax(&act->masked_scores[i * row], &act->probs[i * row], row) < 0)
			return -1;

	// 7. 확률(probs)과 Value(V)를 곱하고 출력 가중치(Wo)로 선형 투영
	if (matmul(act->probs, act->V, act->attn_out, row, row, dim) < 0)
		return -1;
	if (matmul(act->attn_out, w->Wo, act->proj_out, row, dim, dim) < 0)
		return -1;
	
	// 8. 어텐션 출력 후 정규화(Post-Attention RMSNorm) 및 첫 번째 잔차 연결(X + norm_att_out)
	if (rmsnorm(act->proj_out, w->gamma_att_out, act->norm_att_out, row, dim) < 0)
		return -1;
	if (matadd(act->X, act->norm_att_out, act->res_att, row, dim) < 0)
		return -1;

	return 0;
}

// FFN(GeGLU 피드포워드) 서브 블록 순전파: 어텐션 출력(res_att) -> Pre-RMSNorm -> Gate & Up 행렬곱
// -> GELU 활성화 -> Hadamard 곱(문지기 결합) -> Down 행렬곱 -> Post-RMSNorm -> 잔차 연결(res_ffn)
int ffn(Config* cfg, Weights* w, Activations* act)
{
	if (cfg == NULL || w == NULL || act == NULL)
		return -1;
	if (cfg->seq_len <= 0 || cfg->dim <= 0 || cfg->hidden_dim <= 0)
		return -1;

	int row = cfg->seq_len, dim = cfg->dim, h_dim = cfg->hidden_dim;

	// 1. FFN 입력 전 정규화 (Pre-FFN RMSNorm)
	if (rmsnorm(act->res_att, w->gamma_ffn_in, act->norm_ffn_in, row, dim) < 0)
		return -1;

	// 2. Gate와 Up 두 갈래로 차원 확장 (dim -> hidden_dim)
	if (matmul(act->norm_ffn_in, w->W_gate, act->gate, row, dim, h_dim) < 0)
		return -1;
	if (matmul(act->norm_ffn_in, w->W_up, act->up, row, dim, h_dim) < 0)
		return -1;

	// 3. Gate에 GELU 활성화 함수를 적용한 뒤 Up과 원소별 곱셈(Hadamard) 수행
	if (gelu(act->gate, act->gate_act, row, h_dim) < 0)
		return -1;
	if (hadamard(act->gate_act, act->up, act->hidden, row, h_dim) < 0)
		return -1;

	// 4. Down 가중치 행렬을 곱해 원래 차원(hidden_dim -> dim)으로 복구
	if (matmul(act->hidden, w->W_down, act->ffn_out, row, h_dim, dim) < 0)
		return -1;

	// 5. FFN 출력 후 정규화(Post-FFN RMSNorm) 및 두 번째 잔차 연결(res_att + norm_ffn_out)
	if (rmsnorm(act->ffn_out, w->gamma_ffn_out, act->norm_ffn_out, row, dim) < 0)
		return -1;
	if (matadd(act->res_att, act->norm_ffn_out, act->res_ffn, row, dim) < 0)
		return -1;
	return 0;
}

// 모델 전체 순전파 (Forward Pass): 입력 토큰 배열(tokens)을 받아 각 위치의 다음 단어 확률 분포(out_probs)까지 계산
int forward(int* tokens, Config* cfg, Weights* w, Activations* act)
{
	if (tokens == NULL || cfg == NULL || w == NULL || act == NULL)
		return -1;
	if (cfg->seq_len <= 0 || cfg->dim <= 0 || cfg->hidden_dim <= 0 || cfg->vocab_size <= 0)
		return -1;

	int row = cfg->seq_len, dim = cfg->dim, vocab = cfg->vocab_size;

	// 1. 토큰 임베딩 조회(Lookup): 각 단어 번호(tokens[i])에 해당하는 벡터를 꺼내 입력 행렬 X에 복사
	for (int i = 0; i < row; i++)
	{
		if (tokens[i] < 0 || tokens[i] >= vocab)
			return -1;
		for (int j = 0; j < dim; j++)
			act->X[i * dim + j] = w->token_embedding[tokens[i] * dim + j];

	}

	// 2. 임베딩 스케일링: Gemma 구조에 맞춰 임베딩 벡터에 sqrt(dim)을 곱해 신호 크기를 증폭
	if (scalemul(act->X, act->X, sqrt(dim), row, dim) < 0)
		return -1;

	// 3. 어텐션(Attention) 블록 및 FFN 블록 순차 통과
	if (attention(cfg, w, act) < 0)
		return -1;
	if (ffn(cfg, w, act) < 0)
		return -1;

	// 4. 최종 출력 정규화(Final RMSNorm) 및 단어장 크기(vocab)로 선형 변환(raw_logits)
	if (rmsnorm(act->res_ffn, w->final_gamma, act->final_norm, row, dim) < 0)
		return -1;
	if (matmul(act->final_norm, w->W_out, act->raw_logits, row, dim, vocab) < 0)
		return -1;

	// 5. 최종 로짓 Softcap(30.0) 적용 및 각 토큰 위치별 Softmax 확률 분포(out_probs) 산출
	if (softcap(act->raw_logits, act->logits, 30.0, row, vocab) < 0)
		return -1;
	for (int i = 0; i < row; i++)
		if (softmax(&act->logits[i * vocab], &act->out_probs[i * vocab], vocab) < 0)
			return -1;

	return 0;
}

/* ============================================================================
 * [6] 트랜스포머 블록 및 전체 모델 역전파(Backward) 조립 함수 (3개)
 * ============================================================================ */

// FFN 서브 블록 역전파: d_res_ffn으로부터 역순으로 미분하여 FFN 가중치 기울기들과 d_res_att를 계산
int ffn_backward(Config* cfg, Weights* w, Activations* act, Gradients* grad)
{
	if (cfg == NULL || w == NULL || act == NULL || grad == NULL)
		return -1;
	if (cfg->seq_len <= 0 || cfg->dim <= 0 || cfg->hidden_dim <= 0)
		return -1;

	int row = cfg->seq_len, dim = cfg->dim, h_dim = cfg->hidden_dim;

	// 1. FFN 출력 후 정규화(Post-FFN RMSNorm) 역전파
	if (rmsnorm_backward(act->ffn_out, w->gamma_ffn_out, grad->d_res_ffn, grad->d_ffn_out, grad->d_gamma_ffn_out, row, dim) < 0)
		return -1;

	// 2. Down 선형 변환(ffn_out = hidden * W_down) 역전파: d_W_down 및 d_hidden 계산
	if (transpose(act->hidden, grad->temp1, row, h_dim) < 0)
		return -1;
	if (matmul(grad->temp1, grad->d_ffn_out, grad->d_W_down, h_dim, row, dim) < 0)
		return -1;

	if (transpose(w->W_down, grad->temp1, h_dim, dim) < 0)
		return -1;
	if (matmul(grad->d_ffn_out, grad->temp1, grad->d_hidden, row, dim, h_dim) < 0)
		return -1;

	// 3. Hadamard 곱(hidden = gate_act * up) 및 GELU 활성화 역전파
	if (hadamard(grad->d_hidden, act->up, grad->d_gate_act, row, h_dim) < 0)
		return -1;
	if (hadamard(grad->d_hidden, act->gate_act, grad->d_up, row, h_dim) < 0)
		return -1;

	if (gelu_backward(act->gate, grad->d_gate_act, grad->d_gate, row, h_dim) < 0)
		return -1;

	// 4. Gate & Up 가중치 기울기(d_W_gate, d_W_up) 계산
	if (transpose(act->norm_ffn_in, grad->temp1, row, dim) < 0)
		return -1;
	if (matmul(grad->temp1, grad->d_gate, grad->d_W_gate, dim, row, h_dim) < 0)
		return -1;
	if (matmul(grad->temp1, grad->d_up, grad->d_W_up, dim, row, h_dim) < 0)
		return -1;

	// 5. Gate와 Up 두 갈래에서 내려온 기울기를 합산하여 d_norm_ffn_in 계산
	if (transpose(w->W_gate, grad->temp1, dim, h_dim) < 0)
		return -1;
	if (matmul(grad->d_gate, grad->temp1, grad->d_norm_ffn_in, row, h_dim, dim) < 0)
		return -1;
	if (transpose(w->W_up, grad->temp1, dim, h_dim) < 0)
		return -1;
	if (matmul(grad->d_up, grad->temp1, grad->temp2, row, h_dim, dim) < 0)
		return -1;
	if (matadd(grad->d_norm_ffn_in, grad->temp2, grad->d_norm_ffn_in, row, dim) < 0)
		return -1;

	// 6. FFN 입력 전 정규화(Pre-FFN RMSNorm) 역전파 및 잔차 연결 기울기(d_res_ffn) 합산 -> d_res_att 완성
	if (rmsnorm_backward(act->res_att, w->gamma_ffn_in, grad->d_norm_ffn_in, grad->temp1, grad->d_gamma_ffn_in, row, dim) < 0)
		return -1;
	if (matadd(grad->d_res_ffn, grad->temp1, grad->d_res_att, row, dim) < 0)
		return -1;
	return 0;
}

// 어텐션 서브 블록 역전파: d_res_att로부터 역순으로 미분하여 어텐션 가중치 기울기들과 d_X를 계산
int attention_backward(Config* cfg, Weights* w, Activations* act, Gradients* grad)
{
	if (cfg == NULL || w == NULL || act == NULL || grad == NULL)
		return -1;
	if (cfg->seq_len <= 0 || cfg->dim <= 0 || cfg->window_size <= 0)
		return -1;

	int row = cfg->seq_len, dim = cfg->dim;
	
	// 1. 어텐션 출력 후 정규화(Post-Attention RMSNorm) 역전파
	if (rmsnorm_backward(act->proj_out, w->gamma_att_out, grad->d_res_att, grad->d_proj_out, grad->d_gamma_att_out, row, dim) < 0)
		return -1;
	
	// 2. 출력 투영(proj_out = attn_out * Wo) 역전파: d_Wo 및 d_attn_out 계산
	if (transpose(act->attn_out, grad->temp1, row, dim) < 0)
		return -1;
	if (matmul(grad->temp1, grad->d_proj_out, grad->d_Wo, dim, row, dim) < 0)
		return -1;

	if (transpose(w->Wo, grad->temp1, dim, dim) < 0)
		return -1;
	if (matmul(grad->d_proj_out, grad->temp1, grad->d_attn_out, row, dim, dim) < 0)
		return -1;

	// 3. 문맥 결합(attn_out = probs * V) 역전파: d_V 및 d_probs 계산
	if (transpose(act->probs, grad->temp1, row, row) < 0)
		return -1;
	if (matmul(grad->temp1, grad->d_attn_out, grad->d_V, row, row, dim) < 0)
		return -1;
	if (transpose(act->V, grad->temp1, row, dim) < 0)
		return -1;
	if (matmul(grad->d_attn_out, grad->temp1, grad->d_probs, row, dim, row) < 0)
		return -1;

	// 4. 행별 Softmax 역전파, 어텐션 Softcap(50.0) 역전파, 1/sqrt(dim) 스케일링 역전파
	for (int i = 0; i < row; i++)
		if (softmax_backward(&act->probs[i * row], &grad->d_probs[i * row], &grad->d_scores[i * row], row) < 0)
			return -1;

	if (softcap_backward(act->scores, grad->d_scores, grad->d_scores, 50.0, row, row) < 0)
		return -1;
	if (scalemul(grad->d_scores, grad->d_scores, 1.0 / sqrt(dim), row, row) < 0)
		return -1;

	// 5. 어텐션 점수(scores = Q_rope * K_rope^T) 역전파: d_Q_rope 및 d_K_rope 계산
	if (matmul(grad->d_scores, act->K_rope ,grad->d_Q_rope, row, row, dim) < 0)
		return -1;
	
	if (transpose(grad->d_scores, grad->temp1, row, row) < 0)
		return -1;
	if (matmul(grad->temp1, act->Q_rope, grad->d_K_rope, row, row, dim) < 0)
		return -1;

	// 6. Query와 Key의 회전 위치 인코딩(RoPE) 역전파
	if (rope_backward(grad->d_Q_rope, grad->d_Q, row, dim) < 0)
		return -1;
	if (rope_backward(grad->d_K_rope, grad->d_K, row, dim) < 0)
		return -1;

	// 7. Wq, Wk, Wv 가중치 기울기(d_Wq, d_Wk, d_Wv) 계산
	if (transpose(act->norm_att_in, grad->temp1, row, dim) < 0)
		return -1;
	if (matmul(grad->temp1, grad->d_Q, grad->d_Wq, dim, row, dim) < 0)
		return -1;
	if (matmul(grad->temp1, grad->d_K, grad->d_Wk, dim, row, dim) < 0)
		return -1;
	if (matmul(grad->temp1, grad->d_V, grad->d_Wv, dim, row, dim) < 0)
		return -1;

	// 8. Q, K, V 세 갈래에서 내려온 기울기를 모두 합산하여 d_norm_att_in 계산
	if (transpose(w->Wq, grad->temp1, dim, dim) < 0)
		return -1;
	if (matmul(grad->d_Q, grad->temp1, grad->d_norm_att_in, row, dim, dim) < 0)
		return -1;

	if (transpose(w->Wk, grad->temp1, dim, dim) < 0)
		return -1;
	if (matmul(grad->d_K, grad->temp1, grad->temp2, row, dim, dim) < 0)
		return -1;
	if (matadd(grad->d_norm_att_in, grad->temp2, grad->d_norm_att_in, row, dim) < 0)
		return -1;

	if (transpose(w->Wv, grad->temp1, dim, dim) < 0)
		return -1;
	if (matmul(grad->d_V, grad->temp1, grad->temp2, row, dim, dim) < 0)
		return -1;
	if (matadd(grad->d_norm_att_in, grad->temp2, grad->d_norm_att_in, row, dim) < 0)
		return -1;

	// 9. 어텐션 입력 전 정규화(Pre-Attention RMSNorm) 역전파 및 첫 번째 잔차 연결 기울기 합산 -> d_X 완성
	if (rmsnorm_backward(act->X, w->gamma_att_in, grad-> d_norm_att_in, grad->temp1, grad->d_gamma_att_in, row, dim) < 0)
		return -1;
	if (matadd(grad->d_res_att, grad->temp1, grad->d_X, row, dim) < 0)
		return -1;
	return 0;
}

// 모델 전체 역전파 (Backward Pass): 정답 토큰(targets)과의 오차로부터 출발해 모든 가중치의 기울기(Gradients)를 계산
int backward(int* tokens, int* targets, Config* cfg, Weights* w, Activations* act, Gradients* grad)
{
	if(tokens == NULL || targets == NULL || cfg == NULL || w == NULL || act == NULL || grad == NULL)
		return -1;
	if (cfg->seq_len <= 0 || cfg->dim <= 0 || cfg->hidden_dim <= 0 || cfg->vocab_size <= 0)
		return -1;

	int row = cfg->seq_len, dim = cfg->dim, vocab = cfg->vocab_size;

	// 1. 누적(+=) 방식으로 계산되는 기울기 버퍼들(임베딩 및 5개 RMSNorm gamma)을 먼저 0.0으로 초기화
	if (zero_grad(grad->d_token_embedding, vocab, dim) < 0)
		return -1;
	if (zero_grad(grad->d_final_gamma, 1, dim) < 0)
		return -1;
	if (zero_grad(grad->d_gamma_att_in, 1, dim) < 0)
		return -1;
	if (zero_grad(grad->d_gamma_att_out, 1, dim) < 0)
		return -1;
	if (zero_grad(grad->d_gamma_ffn_in, 1, dim) < 0)
		return -1;
	if (zero_grad(grad->d_gamma_ffn_out, 1, dim) < 0)
		return -1;

	// 2. 출력층 역전파: Cross-Entropy + Softmax 미분 -> 최종 Softcap(30.0) 미분
	for (int i  = 0; i < row; i++)
		if(cross_entropy_backward(&act->out_probs[i * vocab], &grad->d_logits[i * vocab], targets[i], vocab) < 0)
			return -1;

	if (softcap_backward(act->raw_logits, grad->d_logits, grad->d_raw_logits, 30.0, row, vocab) < 0)
		return -1;

	// 3. 출력 가중치(W_out)의 기울기(d_W_out) 및 d_final_norm 계산
	if (transpose(act->final_norm, grad->temp1, row, dim) < 0)
		return -1;
	if (matmul(grad->temp1, grad->d_raw_logits, grad->d_W_out, dim, row, vocab) < 0)
		return -1;

	if (transpose(w->W_out, grad->temp1, dim, vocab) < 0)
		return -1;
	if (matmul(grad->d_raw_logits, grad->temp1,grad->d_final_norm, row,  vocab, dim) < 0)
		return -1;

	// 4. 최종 RMSNorm 역전파 -> FFN 블록 역전파 -> 어텐션 블록 역전파 순차 호출
	if (rmsnorm_backward(act->res_ffn, w->final_gamma, grad->d_final_norm, grad->d_res_ffn, grad->d_final_gamma, row, dim) < 0)
		return -1;

	if (ffn_backward(cfg, w, act, grad) < 0)
		return -1;

	if (attention_backward(cfg, w, act, grad) < 0)
		return -1;

	// 5. 임베딩 sqrt(dim) 스케일링 역전파 및 입력된 단어 번호(tokens[i]) 위치의 임베딩 기울기 누적
	if (scalemul(grad->d_X, grad->d_X, sqrt(dim), row, dim) < 0)
		return -1;

	for (int i = 0; i < row; i++)
		for(int j = 0; j < dim; j++)
			grad->d_token_embedding[tokens[i] * dim + j] += grad->d_X[i * dim + j];
	return 0;
}

/* ============================================================================
 * [7] 모델 전체 가중치 업데이트 및 메모리 할당/초기화 함수 (2개)
 * ============================================================================ */

// 전체 가중치 업데이트: 역전파로 구한 14개 가중치의 기울기(grad)를 사용해 모든 가중치(w)를 일괄 업데이트
int update_weights (Config* cfg, Weights* w, Gradients* grad, double lr)
{
	if (cfg == NULL || w == NULL || grad == NULL || lr <= 0.0)
		return -1;
	if (cfg->seq_len <= 0 || cfg->dim <= 0 || cfg->hidden_dim <= 0 || cfg->vocab_size <= 0)
		return -1;

	int dim = cfg->dim, h_dim = cfg->hidden_dim, vocab = cfg->vocab_size;

	// 1. 입출력 가중치 3개 업데이트
	if (sgd_update(w->token_embedding, grad->d_token_embedding, lr, vocab, dim) < 0)
		return -1;
	if (sgd_update(w->final_gamma, grad->d_final_gamma, lr, 1, dim) < 0)
		return -1;
	if (sgd_update(w->W_out, grad->d_W_out, lr, dim, vocab) < 0)
		return -1;

	// 2. 어텐션 가중치 4개 업데이트
	if (sgd_update(w->Wq, grad->d_Wq, lr, dim, dim) < 0)
		return -1;
	if (sgd_update(w->Wk, grad->d_Wk, lr, dim, dim) < 0)
		return -1;
	if (sgd_update(w->Wv, grad->d_Wv, lr, dim, dim) < 0)
		return -1;
	if (sgd_update(w->Wo, grad->d_Wo, lr, dim, dim) < 0)
		return -1;

	// 3. FFN 가중치 3개 업데이트
	if (sgd_update(w->W_gate, grad->d_W_gate, lr, dim, h_dim) < 0)
		return -1;
	if (sgd_update(w->W_up, grad->d_W_up, lr, dim, h_dim) < 0)
		return -1;
	if (sgd_update(w->W_down, grad->d_W_down, lr, h_dim, dim) < 0)
		return -1;

	// 4. 서브 블록 앞뒤 RMSNorm 배율 가중치 4개 업데이트
	if (sgd_update(w->gamma_att_in, grad->d_gamma_att_in, lr, 1, dim) < 0)
		return -1;
	if (sgd_update(w->gamma_att_out, grad->d_gamma_att_out, lr, 1, dim) < 0)
		return -1;
	if (sgd_update(w->gamma_ffn_in, grad->d_gamma_ffn_in, lr, 1, dim) < 0)
		return -1;
	if (sgd_update(w->gamma_ffn_out, grad->d_gamma_ffn_out, lr, 1, dim) < 0)
		return -1;
	return 0;
}

// 모델 초기화: Weights, Activations, Gradients 구조체의 모든 포인터 메모리를 할당(calloc)하고 가중치 초기값 설정
int init_model(Config* cfg, Weights* w, Activations* act, Gradients* grad)
{
	if (cfg == NULL || w == NULL || act == NULL || grad == NULL)
		return -1;
	if (cfg->seq_len <= 0 || cfg->dim <= 0 || cfg->hidden_dim <= 0 || cfg->vocab_size <= 0)
		return -1;

	int row = cfg->seq_len , dim = cfg->dim, h_dim = cfg->hidden_dim, vocab = cfg->vocab_size;
	int max_dim = row + dim + h_dim + vocab;

	// 1. Weights(가중치) 14개 메모리 할당 및 난수/1.0 초기화
	w->token_embedding = calloc(vocab * dim, sizeof(double));
	w->final_gamma = calloc(dim, sizeof(double));
	w->W_out = calloc(dim * vocab, sizeof(double));

	w->Wq = calloc(dim * dim, sizeof(double));
	w->Wk = calloc(dim * dim, sizeof(double));
	w->Wv = calloc(dim * dim, sizeof(double));
	w->Wo = calloc(dim * dim, sizeof(double));

	w->W_gate = calloc(dim * h_dim, sizeof(double));
	w->W_up = calloc(dim * h_dim, sizeof(double));
	w->W_down = calloc(h_dim * dim, sizeof(double));

	w->gamma_att_in = calloc(dim, sizeof(double));
	w->gamma_att_out = calloc(dim, sizeof(double));
	w->gamma_ffn_in = calloc(dim, sizeof(double));
	w->gamma_ffn_out = calloc(dim, sizeof(double));

	if (init_weight(w->token_embedding, vocab, dim) < 0) return -1;
	if (init_gamma(w->final_gamma, dim) < 0) return -1;
	if (init_weight(w->W_out, dim, vocab) < 0) return -1;

	if (init_weight(w->Wq, dim, dim) < 0) return -1;
	if (init_weight(w->Wk, dim, dim) < 0) return -1;
	if (init_weight(w->Wv, dim, dim) < 0) return -1;
	if (init_weight(w->Wo, dim, dim) < 0) return -1;

	if (init_weight(w->W_gate, dim, h_dim) < 0) return -1;
	if (init_weight(w->W_up, dim, h_dim) < 0) return -1;
	if (init_weight(w->W_down, h_dim, dim) < 0) return -1;

	if (init_gamma(w->gamma_att_in, dim) < 0) return -1;
	if (init_gamma(w->gamma_att_out, dim) < 0) return -1;
	if (init_gamma(w->gamma_ffn_in, dim) < 0) return -1;
	if (init_gamma(w->gamma_ffn_out, dim) < 0) return -1;

	// 2. Activations(순전파 중간 결과 버퍼) 28개 메모리 할당 (0.0으로 자동 초기화)
	act->X = calloc(row * dim, sizeof(double));
	act->norm_att_in = calloc(row * dim, sizeof(double));
	act->Q = calloc(row * dim, sizeof(double));
	act->K = calloc(row * dim, sizeof(double));
	act->V = calloc(row * dim, sizeof(double));
	act->Q_rope = calloc(row * dim, sizeof(double));
	act->K_rope = calloc(row* dim , sizeof(double));
	act->K_T = calloc(dim * row, sizeof(double));
	act->scores = calloc(row * row, sizeof(double));
	act->capped_scores = calloc(row * row, sizeof(double));
	act->masked_scores = calloc(row * row, sizeof(double));
	act->probs = calloc(row * row, sizeof(double));
	act->attn_out = calloc(row * dim, sizeof(double));
	act->proj_out = calloc(row * dim, sizeof(double));
	act->norm_att_out = calloc(row * dim, sizeof(double));
	act->res_att = calloc(row * dim, sizeof(double));

	act->norm_ffn_in = calloc(row * dim, sizeof(double));
	act->gate = calloc(row * h_dim, sizeof(double));
	act->gate_act = calloc(row * h_dim, sizeof(double));
	act->up = calloc(row * h_dim, sizeof(double));
	act->hidden = calloc(row * h_dim, sizeof(double));
	act->ffn_out = calloc(row * dim, sizeof(double));
	act->norm_ffn_out = calloc(row * dim, sizeof(double));
	act->res_ffn = calloc(row * dim, sizeof(double));

	act->final_norm = calloc(row * dim, sizeof(double));
	act->raw_logits = calloc(row * vocab, sizeof(double));
	act->logits = calloc(row * vocab, sizeof(double));
	act->out_probs = calloc(row * vocab, sizeof(double));

	// 3. Gradients(역전파 기울기 버퍼) 40개 메모리 할당 (0.0으로 자동 초기화)
	grad->d_token_embedding = calloc(vocab * dim, sizeof(double));
	grad->d_final_gamma = calloc(dim, sizeof(double));
	grad->d_W_out = calloc(dim * vocab, sizeof(double));

	grad->d_Wq = calloc(dim * dim, sizeof(double));
	grad->d_Wk = calloc(dim * dim, sizeof(double));
	grad->d_Wv = calloc(dim * dim, sizeof(double));
	grad->d_Wo = calloc(dim * dim, sizeof(double));

	grad->d_W_gate = calloc(dim * h_dim, sizeof(double));
	grad->d_W_up = calloc(dim * h_dim, sizeof(double));
	grad->d_W_down = calloc(h_dim * dim, sizeof(double));

	grad->d_gamma_att_in = calloc(dim, sizeof(double));
	grad->d_gamma_att_out = calloc(dim, sizeof(double));
	grad->d_gamma_ffn_in = calloc(dim, sizeof(double));
	grad->d_gamma_ffn_out = calloc(dim, sizeof(double));

	grad->d_logits = calloc(row * vocab, sizeof(double));
	grad->d_raw_logits = calloc(row * vocab, sizeof(double));
	grad->d_final_norm = calloc(row * dim, sizeof(double));
	grad->d_res_ffn = calloc(row * dim, sizeof(double));

	grad->d_norm_ffn_out = calloc(row * dim, sizeof(double));
	grad->d_ffn_out = calloc(row * dim, sizeof(double));
	grad->d_hidden = calloc(row * h_dim, sizeof(double));
	grad->d_gate_act = calloc(row * h_dim, sizeof(double));
	grad->d_gate = calloc(row * h_dim, sizeof(double));
	grad->d_up = calloc(row * h_dim, sizeof(double));
	grad->d_norm_ffn_in = calloc(row * dim, sizeof(double));
	grad->d_res_att = calloc(row * dim, sizeof(double));

	grad->d_norm_att_out = calloc(row * dim, sizeof(double));
	grad->d_proj_out = calloc(row * dim, sizeof(double));
	grad->d_attn_out = calloc(row * dim, sizeof(double));
	grad->d_probs = calloc(row * row, sizeof(double));
	grad->d_scores = calloc(row * row, sizeof(double));
	grad->d_Q_rope = calloc(row * dim, sizeof(double));
	grad->d_K_rope = calloc(row * dim, sizeof(double));
	grad->d_Q = calloc(row * dim, sizeof(double));
	grad->d_K = calloc(row * dim, sizeof(double));
	grad->d_V = calloc(row * dim, sizeof(double));
	grad->d_norm_att_in = calloc(row * dim, sizeof(double));
	grad->d_X = calloc(row * dim, sizeof(double));

	grad->temp1 = calloc(max_dim * max_dim, sizeof(double));
	grad->temp2 = calloc(max_dim * max_dim, sizeof(double));

	return 0;
}

