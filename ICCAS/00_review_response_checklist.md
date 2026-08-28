# ICCAS 2026 Camera-Ready — 리뷰 반영 체크리스트

- **논문**: *Comparative Study on Agility, Efficiency, and Impact Absorption of Bipedal Robots with Active Toes*
- **마감**: 2026-08-31 / **상태**: 전 항목 반영 완료, 6페이지 수납 확인 (2026-08-28)
- 코멘트 출처: Associate Editor(AE), 리뷰어1(R1), 리뷰어2(R2)

## 반영한 코멘트

| ✓ | 사안 | 리뷰어 | 반영 내용 |
|---|---|---|---|
| ✅ | agility 주장 완화 | AE, R1, R2 | "confirms improved turning agility" 등 삭제. 속도 −4.4%를 Abstract·결론에 명시, 결과를 경로 편차 감소 관찰로 한정 |
| ✅ | high-fidelity / sim-to-real 표현 | AE, R1, R2 | "high-fidelity"→"actuator- and transmission-aware" (Abstract·키워드·결론). 절 제목·본문의 "gap 최소화" 주장 제거, 미검증 파라미터(관성·백래시·벨트 강성·접촉 모델)를 결론에 나열 |
| ✅ | CoT 재현 불가 (125 W ↔ 0.316) | AE, R2 | 125 W → **289 W of metabolic power** (Neumann Fig. 15.26 기반, 289/(70·9.81·1.33)=0.316 재현됨). Eq. (5)에 회생 효율 반영, Table 3 "Mechanical loss" 정의를 표 노트에 추가 |
| ✅ | GRF 정의 불일치 | AE, R2 | 4곳 모두 "Heel-strike GRF"로 통일. 본문에 "(resultant magnitude)", 표 레이블에 "(avg. peak)" 명시 |
| ✅ | 수식·기호·참고문헌 | AE, R2 | Eq.(6)→(5) 수정. v_m=0.1 rad/s, σ_arr=0.1 m, d*, [x]⁺, τ_cont, h_th=0.5 전부 정의. Table 1 관절 ω_max를 관절측 값으로 교체. C_p 단위 [W/(Nm)²] 추가. 미인용 ref [24] 삭제·재번호 |
| ✅ | toe-ablation 설정 불명확 | AE, R1 | Sec. III-B에 명세: toe 관절을 중립 0° 고정 관절로 교체, 링크·질량·관성·접촉 형상 유지(총질량 32.0 kg 불변), 구동 DOF만 제거 (자코비안 7×7→6×6) |
| ✅ | active toe 서술 부족 | R2 | [20] 무변경 계승 명시 + Eq. (3) toe 열 해설(toe 모터의 knee·ankle 토크 기여). 기하 치수는 Fig. 2(a)가 담당 |
| ✅ | 문체·용어·약어 | R2 | "rigorously"·"significantly"·"unbiased setup" 등 제거. biped→bipedal 통일. CoT·GRF 첫 등장 시 풀네임 정의 |
| ✅ | 그림 보강 | R2 | Fig. 6 캡션에 음영=±1 SD 명시. Fig. 7 캡션에 웨이포인트 순서·궤적 색 스케일(편차 0→0.5 m) 명시 |
| ✅ | 시드 수·변동성 공개 | AE, R1, R2 | Sec. IV-A에 명시: 구성당 단일 시드·단일 정책, 10 trial은 반복 롤아웃, 표 값은 평균. (trial별 std는 원본 로그 소실로 미보고) |
| ✅ | 선회 메커니즘 분석 | AE, R1, R2 | 신규 지표 불가(재실험 없음) → R2의 대안 조건 충족: 결론을 "경로 편차 감소" 관찰로 한정 |

## 리뷰 외 자체 수정

- **Eq. (1)–(3) 모터/관절 방향 반전 수정** — 감속기가 증속기로 기술돼 있던 오류 (행렬 숫자는 유지, 기호만 교환)
- Fig. 5 파일명 오타(컴파일 실패) 수정
- 저자 6인·소속 3곳·ACKNOWLEDGMENT를 제출본대로 복원
- 분량 조절: 로드맵 문단 삭제, 중복 수치·요약 문장 정리, 참고문헌 압축 → 6페이지 수납
