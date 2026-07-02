# Repository Agent Instructions

## Git workflow

- When updating this branch against the target/base branch, use rebase rather than merge.
- Prefer `git fetch origin` followed by `git rebase origin/main` (or the current PR base branch if it is not `main`).
- Do not create merge commits just to synchronize with the base branch.
- If a rebased branch has already been pushed, update it with `git push --force-with-lease` rather than `git push --force`.

## 沟通语言

- 日常沟通必须使用中文，直奔主题，优先说明结论、改动和验证结果。
- 技术报错、异常名、API 名称、命令输出、库名、配置键、HTTP 状态码等保留英文原文。
- 如果出现错误，不要只翻译或概括；必须保留关键英文错误信息，方便用户复制到搜索引擎查询。
- 解释技术方案时可以中文为主，英文术语保留原名，例如 `ECharts`、`localStorage`、`setInterval`、`DOMContentLoaded`。
- 用户英文基础较弱，避免要求用户用英文补充需求；需要确认时用中文问一个关键问题即可。

## 编码与文件规范

- 单个文件不得超过800行代码。
