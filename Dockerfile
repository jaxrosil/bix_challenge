# Use a imagem base do Python
FROM python:3.8-slim

# Defina o diretório de trabalho no container
WORKDIR /app

# Copie os arquivos requirements.txt para o diretório de trabalho
COPY requirements.txt .

# Instale as dependências listadas no requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copie todo o conteúdo do diretório atual para o diretório de trabalho
COPY . .

# Exponha a porta que o Gradio usa
EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"

# Defina o comando padrão para iniciar o app
CMD ["python", "app.py"]
