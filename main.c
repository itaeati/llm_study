#include "llm.h"

int main(void)
{
	// 실행할 때마다 동일한 난수 초기값이 나오도록 시드(Seed) 고정
	srand(42);

	// 1. 트랜스포머 하이퍼파라미터 설정 및 핵심 구조체 선언
	//    - vocab_size = 5  : 단어장 크기 (5개 단어)
	//    - dim = 8         : 각 단어를 8차원 벡터로 표현
	//    - hidden_dim = 16 : FFN 내부에서 16차원으로 확장
	//    - seq_len = 4     : 입력 문장 길이 (토큰 4개)
	//    - window_size = 4 : 슬라이딩 윈도우 범위 (4개)
	Config cfg = {.vocab_size = 5, .dim = 8, .hidden_dim = 16, .seq_len = 4, .window_size = 4};
	Weights w;
	Activations act;
	Gradients grad;

	// 모델의 모든 가중치·활성화값·기울기 버퍼 메모리 할당 및 가중치 초기화
	if (init_model(&cfg, &w, &act, &grad) < 0)
		return -1;

	// 2. 단어장(토크나이저 역할) 및 학습 데이터 준비
	//    - tokens  : 모델에게 보여줄 입력 단어 번호들 ("나는", "오늘", "점심으로", "김치찌개를")
	//    - targets : 각 입력 단어 바로 다음에 나와야 할 정답 단어 번호들 ("오늘", "점심으로", "김치찌개를", "먹었다.")
	const char* words[5] = {"나는", "오늘", "점심으로", "김치찌개를", "먹었다."};
	int tokens[4] = {0, 1, 2, 3};
	int targets[4] = {1, 2, 3, 4};

	// 3. [학습 전 추론] 무작위로 초기화된 상태에서 다음 단어를 어떻게 예측하는지 확인
	printf("=== [학습 전 예측] ===\n");
	forward(tokens, &cfg, &w, &act);
	for (int i = 0; i < cfg.seq_len; i++)
	{
		// i번째 위치에서 가장 확률(out_probs)이 높은 단어 번호를 argmax로 찾음
		int pred = argmax(&act.out_probs[i * cfg.vocab_size], cfg.vocab_size);
		printf("입력: %-12s -> 예측: %-12s (정답: %s)\n", words[tokens[i]], words[pred], words[targets[i]]);
	}

	// 4. [학습 루프] 순전파(Forward) -> 역전파(Backward) -> 가중치 업데이트(SGD)를 200번 반복
	printf("\n=== [학습 진행 중...] ===\n");
	double lr = 0.05; // 학습률 (Learning Rate)
	for (int epoch = 0; epoch <= 200; epoch++)
	{
		// (1) 순전파: 현재 가중치로 각 단어 다음의 확률 분포(out_probs) 계산
		if (forward(tokens, &cfg, &w, &act) < 0) return -1;

		// 40 에폭마다 Cross-Entropy Loss(오차: -log(정답 단어의 예측 확률)) 출력
		if (epoch % 40 == 0)
		{
			double loss = 0.0;
			for (int i = 0; i < cfg.seq_len; i++)
				loss -= log(act.out_probs[i * cfg.vocab_size + targets[i]] + 1e-9);
			printf("Epoch %3d | Loss(오차): %.6f\n", epoch, loss / cfg.seq_len);
		}

		// (2) 역전파: 정답(targets)과의 오차를 거꾸로 전파하며 모든 가중치의 기울기(grad) 계산
		if (backward(tokens, targets, &cfg, &w, &act, &grad) < 0) return -1;

		// (3) 가중치 업데이트: 구한 기울기(grad)와 학습률(lr)로 가중치(w) 수정
		if (update_weights(&cfg, &w, &grad, lr) < 0) return -1;
	}

	// 5. [학습 후 추론] 200번 학습을 마친 모델에게 다시 똑같은 입력을 주어 다음 단어 예측 확인
	printf("\n=== [학습 후 예측] ===\n");
	forward(tokens, &cfg, &w, &act);
	for (int i = 0; i < cfg.seq_len; i++)
	{
		int pred = argmax(&act.out_probs[i * cfg.vocab_size], cfg.vocab_size);
		double prob = act.out_probs[i * cfg.vocab_size + pred] * 100.0;
		printf("입력: %-12s -> 예측: %-12s (확률: %5.1f%% | 정답: %s)\n", words[tokens[i]], words[pred], prob, words[targets[i]]);
	}

	return 0;
}
