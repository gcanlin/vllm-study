---
name: vllm-study-feature-walkthrough
description: Use this skill when writing or updating vllm-study Chinese HTML study notes for vLLM/vLLM-Omni features, especially deep code walkthroughs that need background, usage, full-system diagrams, concrete request debug traces with real shapes, and line-by-line source explanations.
---

# vLLM Study Feature Walkthrough

Use this skill for vLLM Study articles that explain a feature from user-facing usage down to source-level execution. The output should be a Chinese standalone HTML page under `site/`, matching the repo's study-note style.

## Target Article Shape

Write in this order unless the user asks otherwise:

1. **背景与动机**
   - Explain what the feature is, what problem it solves, and why it exists.
   - Call out user-visible behavior vs internal implementation boundaries.
   - Name the exact endpoints, flags, model stages, worker paths, or scheduler features involved.

2. **基本用法**
   - Show how to enable/use the feature.
   - Include concrete commands, HTTP/WebSocket messages, Python calls, env vars, or config fields.
   - Mark the switch that enables streaming, async chunking, parallelism, caching, or other feature-specific behavior.

3. **全景图**
   - Draw a large HTML/SVG/CSS diagram that can occupy most of the screen.
   - Include the major nodes, classes, functions, queues, buffers, tensors, request objects, and output channels.
   - Use arrows to show direction. Prefer orthogonal or clean routed paths; avoid arrows crossing node boxes or text.
   - Do not overfit to one visual style. Use the best representation for the feature: pipeline, swimlanes, stack, matrix/tensor blocks, timing chart, scheduler state machine, etc.
   - Label important node variables and shapes directly on the diagram.

4. **一个具体请求的 debug walkthrough**
   - Pick one representative request and keep it concrete throughout.
   - Show the exact request/client messages first.
   - Use realistic model/config values. If a shape depends on tokenizer, processor resize, runtime batching, TP/PP size, scheduler chunking, or input media dimensions, either compute it from fixed example inputs or explicitly say it is runtime-dependent.
   - Walk downward node by node:
     - Node title should say what happens now.
     - Show exact code statement(s) or a small exact source block for this node. Do not simplify code beyond what is needed for readability.
     - Next to the code, explain the intuition and any non-obvious Python syntax.
     - Show input variables, output variables, dtype, shape, and value examples.
     - Draw a small geometry diagram for the node: tensor matrix, token row, queue, buffer, frame grid, cache map, scheduler state, etc.
     - End with a clear arrow to the next node.
   - If the logic is not serial, keep the main path vertical and open a side column for branches.
   - Prefer finer granularity when unsure. A method that does multiple important things should be split into multiple nodes.

5. **模块源码逐行解析**
   - For each module/class/function involved, first show the concrete method code block and label the class/method.
   - Under the code block, explain it in code order.
   - Do not explain by source line numbers. Say "这一句/这个分支/这个变量" and follow the displayed code order.
   - Include concrete examples and diagrams when shapes, buffers, masks, tokens, communication, cache layout, or scheduler state changes are involved.
   - Keep unrelated refactors or broad architecture commentary out of this section.

6. **总结**
   - State the key boundaries and invariants the reader should remember.
   - Include pitfalls: what is streaming vs not streaming, where data is buffered, what is request scoped, what is engine scoped, what is model scoped.

## Diagram Guidance

- Prefer HTML/CSS/SVG diagrams over ASCII when the article is HTML.
- Diagrams should help the reader feel like they are stepping through a debugger.
- Good diagram content:
  - concrete variable names from code;
  - shape/type/dtype/value examples;
  - queues and buffers before/after mutation;
  - tensor rows/columns and which axis changes;
  - masks and placeholder positions;
  - request ids, scheduler states, stage ids, output event names.
- Avoid diagrams that are only decorative or flat lists of boxes.
- Avoid over-specifying one diagram style in the article. Choose a layout that fits the feature.
- For large diagrams, route arrows through whitespace and avoid crossing boxes or labels.
- For tensor-heavy paths, use matrix/tile visuals similar in spirit to `site/qwen3_structure.html`: dark panels, explicit shape labels, and axis-oriented blocks.

## Accuracy Rules

- Read the current local source before writing. Do not rely on stale PR descriptions if the user asks for current code.
- Use exact class/function names and endpoint names.
- When model constants matter, get them from local config, checked-in docs, or model config. If retrieved externally, mention the source in the article only when useful.
- Distinguish:
  - protocol-level streaming;
  - engine-level streaming input;
  - streaming output;
  - async chunk internal overlap;
  - wire-level async chunk/delta emission.
- If a shape is dynamic, state the formula and the fixed example assumptions.

## Local Workflow

1. Inspect `.claude.md`, `README.md`, and nearby pages for local conventions.
2. Read the source files for endpoints, handlers, preprocessors, model code, scheduler/engine code, and output processors involved.
3. Create/update a standalone Chinese HTML page under `site/`.
4. Update `site/nav.js` and root `README.md` when adding a new page.
5. Validate with:
   - `python3` HTML parser over the changed page;
   - `git diff --check`;
   - targeted `rg` checks for stale PR framing, ASCII-only diagrams, or incorrect line-number references if those are disallowed.
6. Commit only intended files. Do not include `.DS_Store`.
7. Push to `main` when the user asks to commit/push or when `.claude.md`'s commit habit applies and the user has not asked to pause.

## Minimal Article Checklist

- [ ] Background explains why the feature exists.
- [ ] Usage shows the exact entry point and enable switch.
- [ ] Full-system diagram shows main nodes and data flow.
- [ ] One concrete request is walked through end to end.
- [ ] Debug nodes include exact code, explanation, input/output shape, and a geometry diagram.
- [ ] Detailed source walkthrough shows method code first, then code-order explanation.
- [ ] Dynamic shapes are either computed from the example or marked runtime-dependent.
- [ ] HTML and diff checks pass.
