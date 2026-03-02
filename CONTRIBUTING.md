# 贡献指南

感谢您对 VLA Arm Deployment 项目的关注！我们欢迎各种形式的贡献。

## 如何贡献

### 报告问题

如果您发现了 bug 或有功能建议，请通过 GitHub Issues 提交：

1. 检查是否已有相关问题
2. 创建新 issue，使用相应的模板
3. 提供尽可能详细的信息：
   - 问题描述
   - 复现步骤
   - 环境信息（OS, Python 版本, IsaacLab 版本等）
   - 错误日志

### 提交代码

1. **Fork 仓库**
   ```bash
   git clone https://github.com/yourusername/vla-arm-deployment.git
   cd vla-arm-deployment
   ```

2. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   # 或
   git checkout -b fix/your-bug-fix
   ```

3. **进行更改**
   - 遵循现有的代码风格
   - 添加必要的注释
   - 更新相关文档

4. **测试**
   ```bash
   make test
   make lint
   ```

5. **提交更改**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   # 或
   git commit -m "fix: resolve bug description"
   ```

6. **推送到您的 Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **创建 Pull Request**
   - 描述您的更改
   - 关联相关 issue
   - 等待审核

## 代码风格

- 使用 **ruff** 进行代码格式化和检查
- 遵循 PEP 8 规范
- 行长度限制：100 字符
- 使用双引号

```bash
make format  # 格式化代码
make lint    # 检查代码
```

## 提交信息规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

- `feat:` 新功能
- `fix:` 修复 bug
- `docs:` 文档更新
- `style:` 代码格式（不影响功能）
- `refactor:` 重构
- `test:` 测试相关
- `chore:` 构建过程或辅助工具变动

示例：
```
feat: add support for new robot arm
docs: update README with new examples
fix: resolve HDF5 file corruption issue
```

## 目录结构

保持项目结构的清晰：

```
vla-arm-deployment/
├── scripts/          # 可执行脚本
├── src/             # Python 包
├── tasks/           # IsaacLab 任务配置
├── docs/            # 文档
├── tests/           # 测试
└── examples/        # 示例代码
```

## 开发环境设置

```bash
# 创建虚拟环境
conda create -n vla-dev python=3.10
conda activate vla-dev

# 安装开发依赖
pip install -e ".[dev]"

# 安装预提交钩子（推荐）
pre-commit install
```

## 测试

添加新功能时请同时添加测试：

```python
# tests/test_new_feature.py
def test_new_feature():
    # 测试代码
    pass
```

运行测试：
```bash
make test
```

## 文档

更新文档时请确保：

1. 代码注释清晰
2. README.md 保持同步
3. 添加/更新 docs/ 目录下的相关文档

## 许可证

通过提交 PR，您同意您的贡献将在 MIT 许可证下发布。

## 联系方式

如有疑问，请通过 GitHub Issues 或 Discussions 联系我们。

再次感谢您的贡献！