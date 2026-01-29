# Git 上传命令

以下是将更改上传到 Git 仓库的步骤：

1. 确保当前分支是 `main` 分支：
   ```bash
   git checkout main
   ```

2. 拉取最新的远程更改：
   ```bash
   git pull origin main
   ```

3. 添加所有更改到暂存区：
   ```bash
   git add .
   ```

4. 提交更改：
   ```bash
   git commit -m ""
   ```

5. 推送更改到远程仓库：
   ```bash
   git push origin main
   ```