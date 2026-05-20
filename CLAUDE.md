# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## 5. 문서화 규칙

- 중간 과정에서 기록이 필요한 결정/분석/기획은 `dvcc/` 폴더에 `.md`로 추가. 추가시 prefix를 지킬것.
- prefix는 `00_제목.md` — 십의 자리는 그룹묶음, 일의 자리는 그룹내 순서로.
- **매 작업 시작 시 `dvcc/` 디렉토리를 먼저 확인할 것.** 기존 문서 맥락을 반영하고, 내용이 낡았으면 수정, 더 이상 유효하지 않거나 불필요하면 **파일 삭제도 허용**. `dvcc/`는 살아있는 노트 폴더로 관리.
- `README.md`는 사용자용 매뉴얼 — 직접 요청 없으면 수정 금지.
- `dvcc/00_overview.md`의 **불변식** 섹션 변경은 사용자 확인 필수.

## 6. 커밋 타이밍

**중간 커밋을 미루지 말 것. 의미 있는 작업 단위마다 끊어서 커밋.**

- 개발 도중 작업 단위가 의미 있게 마무리되는 시점마다 즉시 커밋해둘 것.
- 나중에 한꺼번에 몰아서 커밋 ❌ — 변경 의도 추적 불가, diff 비대화, 롤백 단위 상실.
- **push 는 나중에 해도 됨** — 로컬 커밋만 쌓아두고 적절한 시점에 일괄 push 가능.
- 메시지 규약·절차 상세는 `dvcc/03_commit_convention.md` 참조.
---