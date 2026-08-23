# ICCAS 2026 Final Submission — 리뷰 대응 할일 인덱스

- **논문**: `ICCAS2026_HyperLeg_RL 20260524_0.tex` — *Comparative Study on Agility, Efficiency,
  and Impact Absorption of Bipedal Robots with Active Toes* (accepted)
- **마감**: 2026-08-31 camera-ready / 작성일 2026-08-24
- **페이지 제한**: 6페이지
- **코멘트**: Associate Editor(7) + 리뷰어1(4) + 리뷰어2(11) = **원 코멘트 22건**
  → 중복 제거 후 **작업 사안 11건**

---

## 1. 할일 인덱스

**위에서부터 순서대로 처리한다.** 
#1~ #8은 본문 수정만으로 끝난다. 
#9~ #11은 새 데이터가 필요하다. 
단 #10·#11이 Table 3·4 수치를 바꾸면 #1·#3·#4가 인용하는 수치도 따라 바뀌므로, 그 세 항목은 마지막에 한 번 더 훑어야 한다.

<table>
<colgroup>
<col style="width:4%">
<col style="width:82%">
<col style="width:14%">
</colgroup>
<thead>
<tr><th>#</th><th>사안</th><th>리뷰어</th></tr>
</thead>
<tbody>
<tr><td><b>1</b></td>
<td>✅ <b>agility 주장 완화.</b> 제목·결론이 "agility"/"turning agility"를 쓰지만 실험은 단일 T 코스이고 개선된 것은 path deviation뿐. toe-equipped는 max·avg 속도가 오히려 낮고 완주 시간은 거의 동일 → speed–accuracy trade-off. path deviation 감소가 선회 역학 개선일 수도, 더 보수적·느린 정책의 결과일 수도 있음. <code>"This confirms that active toes improve turning agility"</code> → <code>"The toe-equipped configuration showed reduced path deviation under the evaluated directional-change task."</code></td>
<td>AE, R1, R2</td></tr>
<tr><td><b>2</b></td>
<td><b>high-fidelity / sim-to-real 표현 정밀화 + 잔여 불확실성 명시.</b> link inertial params, belt elasticity, structural compliance, backlash, contact stiffness/damping, actuator response에 대한 정량적 하드웨어 검증이 없음. 제안 문구: <code>"high-fidelity simulation"</code>→<code>"actuator- and transmission-aware simulation"</code>, <code>"reduces the sim-to-real gap"</code>→<code>"is intended to support future sim-to-real transfer"</code>, <code>"a rigorous foundation for closing the sim-to-real gap"</code>→<code>"a simulation-based evaluation prior to hardware deployment"</code>. 결론에서 시뮬레이션 관찰과 실기 검증을 명확히 구분</td>
<td>AE, R1, R2</td></tr>
<tr><td><b>3</b></td>
<td><b>CoT 정의·재현 가능성.</b> Intro의 "70 kg, 125 W, 1.33 m/s → CoT 0.316"이 재현 불가 — <b>계산 확인: 0.137</b>. 125 W가 metabolic인지 mechanical인지 밝히고 0.316 도출 과정 설명. Table 1은 motoring/regen 효율을 따로 주는데 CoT 식은 signed mechanical power만 써서 regeneration이 battery-side power에 들어가는 방식이 불명. Table 3 "Mechanical loss"의 의미도 명확히</td>
<td>AE, R2</td></tr>
<tr><td><b>4</b></td>
<td><b>GRF 지표 정의 통일.</b> Abstract "heel-strike GRF" vs Table 3 "Avg GRF" — 같은 지표가 아님. 명시할 것: vertical인지 3D 합력인지 / peak인지 평균인지 / 평균 구간이 heel strike인지 stance 전체인지 / 한 발인지 양발인지 / heel strike 검출 방법 / 힘 필터링 여부. 정의 전까지 5.0%를 impact absorption 개선으로 해석 불가</td>
<td>AE, R2</td></tr>
<tr><td><b>5</b></td>
<td><b>수식·기호·참고문헌 정리.</b> ① Table 2가 <b>없는 Eq. (6)을 참조</b> — 논문 수식은 5개뿐이고 CoT는 <b>Eq. (5)</b> (확인됨). ② Table 2 goal-arrival 보상식에 현재 위치 항 누락 의심. ③ <code>v_m</code>, <code>d*</code>, <code>σ_arr</code>, <code>[x]+</code>, <code>τ_cont,i</code> 의미·값 제시 + 모터별 <code>h_th</code> 값 보고. ④ Table 1의 joint <code>ω_max</code>가 motor <code>ω_0</code>와 <b>완전 동일</b>(300.0/327.2/366.0) — Hip 25:1이면 관절은 12 rad/s여야 함 (확인됨). ⑤ <code>C_p</code> 단위·도출 근거. ⑥ <b>ref [24] 미인용</b> — 인용하거나 삭제</td>
<td>AE, R2</td></tr>
<tr><td><b>6</b></td>
<td><b>toe-ablation 설정 명확화.</b> action이 14→12로 줄었으나 toe joint·actuator·link·mass/inertia·contact geometry 중 무엇이 제거/고정/변경됐는지 불명. active toe 구동과 접촉 형상 변화가 <b>둘 다</b> 개선에 기여했을 수 있어 중요</td>
<td>AE, R1</td></tr>
<tr><td><b>7</b></td>
<td><b>기계 설계 절의 active toe 서술 보강.</b> hip/knee/ankle 액추에이터에 지면을 많이 쓰고 핵심인 active toe는 짧고 대부분 [20]으로 미룸. 필요 정보: toe length, mass, RoM, joint-axis location, contact-surface geometry, torque capacity, toe 모터가 ankle·knee 토크에 기여하는 정도, 유효 감속비의 자세 의존성, [20]에서 계승/변경한 것. 일반 액추에이터 서술을 압축해 지면 확보</td>
<td>R2</td></tr>
<tr><td><b>8</b></td>
<td><b>문체·용어·약어.</b> <code>"rigorously isolate"</code>, <code>"ensuring an unbiased setup"</code>, <code>"confirms"</code>, <code>"significantly improves"</code>가 근거보다 강함 → <code>"compare"</code>, <code>"using the same training procedure"</code>, <code>"suggests"</code>, <code>"showed a reduction under the evaluated conditions"</code>. "biped robot"/"bipedal robot" 표기 통일. RL·CoT·GRF 등 약어를 첫 등장 시 정의, 한 번만 쓰는 용어는 약어화 금지</td>
<td>R2</td></tr>
<tr><td><b>9</b></td>
<td><b>그림 보강.</b> Fig. 6 캡션에 음영 영역이 std / standard error / range 중 무엇인지 명시(#10과 연결). Fig. 7에 축 레이블·단위, 기준 경로와 로봇 궤적 구분 표시를 넣어 본문 없이도 이해되게. <b>Fig. 7은 재플롯이 필요해 궤적 데이터에 의존</b></td>
<td>R2</td></tr>
<tr><td><b>10</b></td>
<td><b>학습 시드 수 + 10 trial 변동성(std/range) 보고.</b> Table 3·4가 평균만 제시. 10회가 단일 정책 반복인지 독립 시드인지 명시. Fig. 6 음영 영역의 의미 정의. 정책 1개만 학습했다면 한계 인정. 변동성 분석 없이 "significant" 사용 금지. <b>재학습·재롤아웃 필요</b></td>
<td>AE, R1, R2</td></tr>
<tr><td><b>11</b></td>
<td><b>선회 메커니즘 정량 지표 1~2개 추가.</b> toe는 sagittal 운동인데 왜 yaw 선회·path tracking이 개선되는지 미분석. 후보: yaw-rate tracking error, CoP trajectory, toe-contact duration, stance-foot slip distance, lateral GRF, torso angular-velocity variation, turning radius, lateral acceleration. 분석 없으면 결론을 "path deviation 감소" 관찰에 한정. <b>로거 확장 + 재롤아웃 필요</b></td>
<td>AE, R1, R2</td></tr>
</tbody>
</table>

---

## 2. 지적 강도

| 지적자 수 | 항목 |
|---|---|
| **3명 전원** | **#1** agility 주장, **#2** sim-to-real 표현, **#10** 변동성·시드, **#11** 선회 메커니즘 |
| 2명 | #3 CoT, #4 GRF, #5 수식·기호·참조, #6 toe-ablation 설정 |
| 리뷰어2 단독 | #7 active toe 서술, #8 문체·용어, #9 그림 |

3명이 전원 지적한 **#1·#2·#10·#11** 이 이번 리비전의 핵심이다. 네 항목 모두
"주장이 근거를 초과한다"는 같은 뿌리에서 나왔다 — 리뷰어2 Summary의
*"several claims currently extend beyond what the experiments directly demonstrate"* 가 총평.

---

## 3. 리뷰어별 원 코멘트 대조 (누락 확인용)

| 원 코멘트 | 할일 # |
|---|---|
| AE-1 agility 주장 완화 | 1 |
| AE-2 선회 메커니즘 정량 | 11 |
| AE-3 sim-to-real 표현 정밀화 | 2 |
| AE-4 toe-ablation 설정 | 6 |
| AE-5 변동성 + 시드 수 | 10 |
| AE-6 GRF·CoT 정의 | 3, 4 |
| AE-7 notation·equation·reference | 5 |
| R1-1 toe-ablation 설정 | 6 |
| R1-2 agility 정의 + 선회 지표 | 1, 11 |
| R1-3 실기 실험 없음 / 잔여 불확실성 논의 | 2 |
| R1-4 std·CI, 단일 정책 vs 독립 시드 | 10 |
| R2-3.1 agility 주장 범위 초과 | 1 |
| R2-3.2 선회 메커니즘 미분석 | 11 |
| R2-3.3 CoT 재현 불가 | 3 |
| R2-3.4 GRF 정의 불일치 | 4 |
| R2-3.5 high-fidelity 주장 | 2 |
| R2-3.6 active toe 서술 부족 | 7 |
| R2-3.7 실험 변동성 | 10 |
| R2-4.1 문체·용어·약어 | 8 |
| R2-4.2 수식·기호 | 5 |
| R2-4.3 그림 캡션·축 | 9 |
| R2-4.4 ref [24] 미인용 | 5 |

원 코멘트 22건 전부가 할일 11건에 매핑됨 — 누락 없음.
