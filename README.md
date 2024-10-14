# 小说阅读 AI 助手

## 项目介绍

小说阅读 AI 助手是一个基于 llama-index 实现的 RAG 小说阅读助手，可以向引擎提问小说剧情和人物关系等。
向量嵌入使用的是 [text2vec-base-chinese](https://huggingface.co/shibing624/text2vec-base-chinese)，LLM 模型使用的是 [gpt-3.5-turbo](https://platform.openai.com/docs/models/gpt-3-5)。

## 环境要求

- Python 3.12
- 请参考 `requirements.txt` 安装所需依赖包

## 文件说明

- `app.py`: 主应用程序文件，包含 FastAPI 服务器和聊天功能的实现
- `utils/parse_epub.py`: 用于解析 EPUB 格式电子书的工具
- `utils/embed_novels.py`: 用于将小说内容嵌入向量的工具
- `.env.example`: 环境变量示例文件
- `requirements.txt`: 项目依赖包列表
- `LICENSE`: 项目许可证文件

## 使用步骤

1. 配置环境变量
   复制 `.env.example` 文件并重命名为 `.env`，然后填写必要的配置信息。
   手动创建数据库`novels`

2. 安装依赖包

   ```
   pip install -r requirements.txt
   ```

3. 准备小说数据
   使用 `parse_epub.py` 解析 EPUB 格式的小说文件：

   ```
   python utils/parse_epub.py
   ```

4. 嵌入小说内容
   使用 `embed_novels.py` 将解析后的小说内容嵌入向量：

   ```
   python utils/embed_novels.py
   ```

5. 启动应用
   运行 `app.py` 文件启动 FastAPI 服务器：

   ```
   python app.py
   ```

6. 使用 AI 助手
   服务器启动后，您可以通过 HTTP POST 请求与 AI 助手进行交互。例如：
   ```
   curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"message":"请简要介绍一下这本小说的主要情节"}'
   ```

## 注意事项

- 确保系统有足够的内存和计算资源来处理大型语言模型和向量嵌入。
- 如遇 CUDA 相关错误，请检查 GPU 驱动程序是否与 PyTorch 版本兼容。
- 对于大型小说数据集，嵌入过程可能需要较长时间，请耐心等待。

如有任何问题或需要进一步帮助，请随时询问。祝您使用愉快！
