# ICCAS 2026 Final Submission — 리뷰어 코멘트 대응 체크리스트

- **논문**: `ICCAS2026_HyperLeg_RL 20260524_0.tex` — *Comparative Study on Agility, Efficiency,
  and Impact Absorption of Bipedal Robots with Active Toes* (accepted)
- **마감**: 2026-08-31 camera-ready / 작성일 2026-08-24 → **가용 7일**
- **페이지 제한**: 6페이지
- **수령 코멘트**: Associate Editor, 리뷰어1. **리뷰어2 미수령**
  (AE가 언급한 "notation, equation, and reference issues"의 구체 목록이 리뷰어2에 있을 것으로 추정)

---

## 1. 조사로 확정한 사실

### F1. Toe-ablation은 깨끗한 통제 실험 — 리뷰어1 #1을 완전히 답할 수 있다

`pxr`로 두 USD를 직접 열어 비교:

| 항목 | `HyperLeg.usd` | `HyperLeg_Wo_Toe.usd` |
|---|---|---|
| `L_TO`/`R_TO` | `PhysicsRevoluteJoint` (axis Z, −10~65°) | `PhysicsFixedJoint` |
| 관절 프레임 `pos0/rot0/pos1/rot1` | — | **비트 단위 동일** (중립 0°에서 용접) |
| body 수 / 총질량 | 18 / 32.0537 kg | 18 / 32.0537 kg **동일** |
| toe 링크 `l_to`/`r_to` | 0.2241 kg | **0.2241 kg 유지** |
| 충돌 프림 | 21개 (`l_to`, `l_tp`, `l_heel` 포함) | **21개 동일** |
| CA 자코비안 | 7×7 | 6×6 (TO 행·열 제거) |

→ 질량·관성·링크·**접촉 형상 모두 보존**, 구동 DOF만 제거. 리뷰어가 우려한 교란 요인
(mass/inertia 변화, contact geometry 변화)이 실제로 없다.

근거 코드: `source/hyperleg_rl/assets/hyperleg.py:35,52`,
`source/hyperleg_rl/tasks/manager_based/locomotion/velocity/hyperleg/hyperleg_env_cfg.py:386`

### F2. 체크포인트·로그 확정 소실 → 재학습 필수

`logs/`는 `.gitignore` 대상. wandb 런 디렉토리의 `model_*.pt` **5831개가 전부 끊어진 심볼릭
링크** (`/home/jkkim/IsaacLab/...` → 해당 경로 자체가 없음), 유효 파일 **0개**.
`~/Downloads`의 `.pt` 3개는 action dim 18로 다른 프로젝트.

### F3. wandb `config.yaml`은 실제 파일로 남아 논문 학습 설정을 정확히 복원 가능

논문 Table III·IV를 만든 것으로 추정되는 런 (실행 시각·설정 일치):

| 태스크 | 변형 | wandb 런 |
|---|---|---|
| walking | equipped | `run-20260530_091644-rb6dz0rr` |
| walking | ablation | `run-20260530_110914-tmkzh77m` |
| T-test | equipped | `run-20260613_144806-l8zheh8d` |
| T-test | ablation | `run-20260613_162812-nje6a2ug` |

**hyperleg 런 57개 전부 seed = 42** → 논문은 확정적으로 **단일 시드**.

### F4. 학습 시간 실측 (체크포인트 심볼릭 링크 mtime으로 계산)

| 태스크 | 설정 | 소요 |
|---|---|---|
| walking | 8192 env, 3000 iter | **77 min** (3런 평균 78.6) |
| T-test | 16384 env, 3500 iter | **74 min** (2런) |

가용 자원: **RTX 5090 × 2** (각 32 GB, GPU 1 유휴), 24 core, 125 GB RAM.

### F5. Table II가 실제 학습 설정과 불일치 — 새로 발견한 결함

| 항목 | 논문 Table II | 실제 walking | 실제 T-test |
|---|---|---|---|
| Parallel envs | 8192 | 8192 ✓ | **16384 ✗** |
| Episode | 20 s | 20 s ✓ | **30 s ✗** |
| Iterations | 3000 | 3000 ✓ | **3500 ✗** |
| Learning rate | 1e-4 | 1e-4 ✓ | **1e-3 ✗** |
| Linear velocity (xy) *w* | **+2.5** | **2.0 ✗** | — |
| Angular velocity (z) *w* | **+0.5** | **1.0 ✗** | — |
| Termination *w* | −200 | −200 ✓ | **−75 ✗** |
| Thermal penalty *w* | −10.0 | −10 ✓ | **−5 ✗** |
| Power/CoT 항 | Eq. (6) | `power_consumption` (−1e-4) | `cost_of_transport` (−1e-4) |

논문의 Reward(position-command) 블록은 goal arrival·position progress 2행뿐이나 실제로는
termination·thermal·CoT 3항이 더 있고 walking과 가중치가 다르다. 리포 코드
`agents/rsl_rl_ppo_cfg.py:12`의 `max_iterations = 1000`도 논문 3000과 불일치(CLI 오버라이드였음).

### F6. 선회 메커니즘 지표는 로깅되지 않음

`scripts/rsl_rl/play_T_test.py:151` `_METRICS_COLS`는 `total_time_s`, `seg0~4_s`,
`max/mean_vel_x`, `mean/max_path_dev_m`만 기록. **yaw rate·CoP·측면 가속도·선회 반경 전무.**

### F7. GRF 정의는 코드에 명확히 있다

`source/hyperleg_rl/viz/power_logger.py`의 `_FOOT_GRF_LEFT = ("l_heel",)` — **heel body만**
집계. 즉 Abstract의 "heel-strike GRF"가 맞고 Table III의 "Avg GRF" 레이블이 틀렸다.

---

## 2. 시드를 여러 개 돌려야 하는가

**리뷰어는 다중 시드를 요구하지 않았다.** R1-4는 "std/CI를 보고하고 그 10 trial이 단일 정책의
반복 롤아웃인지 독립 학습 시드인지 **밝혀라**", AE는 "변동성을 학습 시드 수와 **함께** 보고하라".
요구의 본질은 **공개(disclosure)**다. "시드 1개, 단일 정책 10회 롤아웃"이라고 정직하게 적고
std를 붙이면 문자 그대로 충족된다.

**그럼에도 3시드를 권장한다:**

- F2로 재학습이 **어차피 필수**다. 시드 1개든 3개든 파이프라인은 똑같이 돌려야 한다.
- F4 실측으로 3시드 총비용은 **12런 × ~75분 ÷ GPU 2장 ≈ 7.6시간**. 7일 중 하루의 1/3.
- 논문의 가장 약한 지점(n=1)을 거의 공짜로 메꾼다. AE가 시드 수를 물은 것 자체가 이 지점을
  보고 있다는 신호다.

**리스크**: 새 정책의 수치는 논문의 −17.5% CoT 등과 달라진다. 3시드 mean±std로 효과가
축소되거나 agility 방향이 뒤집힐 수도 있다.
**완화**: 시드를 사후 선별하지 않고 **전부 보고**한다. 효과가 사라지면 그대로 쓴다 —
AE가 이미 agility 톤 조정을 요구했으므로 방향 자체는 어긋나지 않는다.

---

## 3. 대응 체크리스트

레인: **A** 텍스트만 / **B** 기존 데이터 재분석 / **C** 코드+재학습+롤아웃 / **D** 기간 내 불가

### C 레인 — 임계 경로, 가장 먼저 착수

- [ ] **W0. 재학습 12런** (W3·W4의 전제)
  - seed 42/43/44 × {walking, T-test} × {equipped, ablation}
  - `max_iterations`가 코드에 1000으로 박혀 있으므로 CLI로 walking 3000 / T-test 3500 지정
  - envs는 F5의 실제값 (walking 8192, T-test 16384). GPU 2장 → 6웨이브 ≈ 7.6 h
  - **verify**: 12개 런에 최종 체크포인트 존재 + **`logs/`를 wandb 심볼릭 링크에 의존하지 않고
    별도 백업** (F2 재발 방지)

- [ ] **W3. 선회 메커니즘 정량 지표 추가** — AE2, R1-2
  - AE는 "yaw-rate error or CoP trajectory", R1은 "turning radius, lateral acceleration,
    or yaw rate" → 교집합이 yaw rate. 두 지표를 넣는다:
    1. **yaw-rate tracking error** [rad/s] — 스칼라, Table IV에 1행 추가
    2. **CoP 전후 이동 범위** [m] — sagittal toe가 왜 turning에 영향을 주는지 설명하는 메커니즘.
       toe가 push-off를 연장해 stance 중 CoP를 더 전방까지 밀어냄 → 선회 시 지지면 활용 증가
  - `play_T_test.py:151` `_METRICS_COLS` + `_TrialMetrics.on_step` 확장. CoP는 기존
    `source/hyperleg_rl/sensors/force_vector_contact_sensor.py`의 접촉력 벡터를 재사용
    (신규 센서 불필요)
  - **verify**: `ttest_trials.csv`에 신규 컬럼이 trial별로 기록되고, toe-equipped의 yaw-rate
    error가 ablation보다 작다는 방향성 확인

- [ ] **W4. 시드 수 + 변동성 보고** — AE5, R1-4
  - 3시드 × 10 trial = config당 30 trial → Table III·IV를 **mean ± std**로 전면 갱신
  - 본문에 "3 independent training seeds × 10 rollouts each" 명시
  - **verify**: 표의 모든 수치에 ±std, Abstract·본문·Conclusion 4곳 수치 일치 (grep 전수)

### A 레인 — 학습 도는 동안 병행

- [ ] **W1. Toe-ablation 설정 명시** — AE4, R1-1. F1을 그대로 서술. Section III-B 첫 문단
  - 초안: *"In the toe-ablation variant, the toe revolute joint is replaced by a fixed joint
    at the neutral (0°) configuration in the identical joint frame; the toe link, its mass
    (0.224 kg per foot), inertia, and all collision geometry are retained, so total mass
    (32.05 kg) and foot–ground contact geometry are unchanged. Only the actuated DOF is
    removed, reducing the CA Jacobian from 7×7 to 6×6."*
  - **verify**: 질량·접촉 형상 불변이 명시적으로 쓰였는지

- [ ] **W2. Agility 주장 톤 조정** — AE1, R1-2
  - "improved turning agility" → "improved path-tracking accuracy under a speed–accuracy
    trade-off". Abstract / Section IV-B 마지막 문장 / Conclusion 3곳. 순증 ~0줄
  - **verify**: agility를 근거 없이 단정하는 문장이 남아있지 않은지 grep

- [ ] **W5. "high-fidelity"/"sim-to-real" 표현 정밀화 + 잔여 불확실성 명시** — AE3, R1-3
  - "high-fidelity simulation" → "actuator- and transmission-aware simulation".
    Abstract / Section III 제목·본문 / Conclusion
  - Conclusion limitation 문단에 belt compliance, backlash, contact-model uncertainty를
    남은 오차원으로 추가. 순증 ~3줄
  - **verify**: 하드웨어 검증 없이 "sim-to-real gap을 줄였다"고 단정하는 문장이 없는지

- [ ] **W6. GRF·CoT 정의 일관화·재현 가능화** — AE6
  - Table III 행 레이블 "Avg GRF [N]" → "Mean heel-strike GRF [N]" (F7 근거). Abstract·
    Conclusion과 용어 통일
  - **Eq. (6)에 드라이버 효율 명시.** 현재 식은 (P_Joule + P_mech)/mgv 뿐이고 η는 Table I에만
    있어 재현 불가. 코드의 실제 형태 (`mech = p<0 ? p·η_regen : p/η_out`, η_out=0.9 /
    η_regen=0.8)를 식에 반영
  - **verify**: 식만 보고 롤아웃 CSV에서 CoT를 재계산할 수 있는지 실제 대조

- [ ] **W7. Table II를 실제 학습 설정과 일치시키기** — AE6 "reproducible". F5 전항목 수정
  - walking / T-test 2열로 분리. 보상 가중치 (lin vel 2.0, ang vel 1.0), T-test의
    envs 16384 · episode 30 s · 3500 iter · lr 1e-3, position-command 블록의 누락된
    termination(−75) · thermal(−5) · CoT(−1e-4) 추가
  - **verify**: Table II의 모든 셀이 재학습 런 `config.yaml`과 일치 (자동 대조 스크립트 권장)

- [ ] **W8. 사전 결함** (코멘트와 무관, 무조건 수정)
  - 그림 파일명 오타 → **컴파일 실패**: `.tex:503`의 `fig05a_gait_compaer_snapshot` →
    `fig05a_gait_compare_snapshot`
  - 미인용 참고문헌 **[24]** (Flayols, humanoid state estimators): 본문 인용처 추가 또는
    삭제 후 [25]~[28] 재번호
  - **Toe RoM 부호 확인**: Table I은 −65~10°, USD는 −10~65°. 부호 규약 확정
    (AE의 "notation issues"에 해당 가능)
  - **verify**: `pdflatex` 3회 무경고, 참고문헌 1~28 전부 본문 등장

### D 레인 — 기간 내 불가, 명시 흡수

- [ ] **W-D1. 실기 하드웨어 검증** — R1-3. 로봇 미제작. Conclusion의 기존 limitation 문단에
  "no hardware validation"을 유지·강화하고 W5의 잔여 불확실성 서술과 묶는다.
  숨기지 않고 명시하는 것이 유일한 선택.

### 대기

- [ ] **W9. 리뷰어2 코멘트 수신 후 추가 트리아지.** AE가 언급한 notation·equation·reference
  이슈의 구체 목록이 여기 있을 것. 도착 시 이 표에 행 추가.

---

## 4. 코멘트 → 작업 매핑

| 출처 | 코멘트 | 작업 | 레인 |
|---|---|---|---|
| AE | agility 주장이 근거를 초과 — path-tracking + speed-accuracy trade-off로 표현 | W2 | A |
| AE | yaw-rate error 또는 CoP 등 메커니즘 정량 1~2개 보고 | W3 | C |
| AE | high-fidelity / sim-to-real 을 actuator·transmission-aware로 정밀화 | W5 | A |
| AE | toe-ablation 설정 명확화 | W1 | A |
| AE | 10 trial 변동성 + 학습 시드 수 보고 | W4 | C |
| AE | GRF·CoT 정의 일관성·재현성 | W6 | A |
| AE | notation·equation·reference 이슈 수정 | W8, W9 | A |
| R1-1 | toe-ablation: 관절/액추에이터/링크/질량/관성/접촉형상 중 무엇이 제거·고정되었는가 | W1 | A |
| R1-2 | agility 정의 명확화 + turning radius / lateral accel / yaw rate 추가 | W2, W3 | A+C |
| R1-3 | 실기 실험 없음 — 최소한 belt compliance·backlash·contact 불확실성 논의 | W5, W-D1 | A+D |
| R1-4 | 10 trial의 std/CI, 단일 정책 반복인가 독립 시드인가 | W4 | C |

---

## 5. 실행 순서

1. **W8 그림 오타 수정 → 컴파일 → 현재 페이지 수 실측**
   (6페이지 여유 확정. 다른 모든 분량 판단의 전제)
2. **W0 재학습 12런 착수** (GPU 2장, 6웨이브 ≈ 7.6 h). 동시에 `logs/` 별도 백업 경로 설정
3. 학습 중 **W3 로거 확장** 구현 (T-test 롤아웃 전까지 완료 필수)
4. 학습 중 **A 레인 W1·W2·W5·W6·W7·W8** 전부 처리
5. 학습 완료 → **롤아웃**: walking `play_goto_x.py --log_csv`,
   T-test `play_T_test.py --log_csv` (3시드 × 2 config × 10 trial)
6. **W4 표 갱신** → 수치를 Abstract / Table / 본문 / Conclusion **4곳 동시** 반영
7. 페이지 초과 시 축소 순서: Section II 액추에이터 서술 → Fig. 2·3 subfigure 수
8. 최종 검증 (§6)

**분량 순증 추정**: W1 4줄 + W3 (표 2행 + 4줄) + W5 3줄 + W6 3줄 + W7 (Table II 확장 ~6줄)
≈ 0.4~0.5 컬럼. 1번에서 실측 후 확정.

**D-day 판정**: 8/28까지 재학습·롤아웃이 끝나지 않으면 W3·W4를 축소(시드 1개 + 공개 서술)하고
A 레인만으로 제출한다.

---

## 6. 최종 검증 (제출 직전, 순서대로)

1. `pdflatex` 3회 — 미해결 참조·미싱 그림·overfull 경고 0
2. 페이지 수 ≤ 6
3. Abstract / Table / 본문 / Conclusion 4곳 수치 일치 — 변경 수치 grep 전수 확인
4. Table II의 모든 셀 ↔ 재학습 런 `config.yaml` 대조
5. Eq. (6)만 보고 롤아웃 CSV에서 CoT 재계산이 되는지 실제 대조 (AE의 "reproducible" 요구)
6. 그림·표·식 번호 참조가 본문 서술과 일치
7. 참고문헌 1~28 전부 인용
8. 이 체크리스트 전 항목 = 완료 또는 D 명시
9. accept 메일의 제출물 목록 (copyright form·저자 등록 등) 확인 — **아직 미확인**
