from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import torch
import torchvision.transforms as transforms

from resnet_modelv2 import CustomResNet50

app = FastAPI()

# Configurar CORS para liberar todas as origens
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas as origens
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos os métodos (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Permitir todos os cabeçalhos
)

# Definir as classes de emoção
classes = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Definir o dispositivo para execução (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar o modelo treinado
modelo_treinado = CustomResNet50(8).get_model()
modelo_treinado.load_state_dict(torch.load('./model2affectnet.pt', map_location=device))
modelo_treinado.eval()  # Colocar o modelo em modo de avaliação
modelo_treinado.to(device)

# Definir transformações da imagem
image_size = 224
transformacao = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Função para prever a emoção
def prever_emocao(image: Image.Image):
    # Converter para RGB se a imagem tiver 4 canais (RGBA)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Aplicar as transformações e fazer a predição
    imagem_transformada = transformacao(image).unsqueeze(0).to(device)
    with torch.no_grad():
        saida = modelo_treinado(imagem_transformada)
        _, pred = torch.max(saida, 1)
        emocao_predita = classes[pred.item()]
    return emocao_predita

# Endpoint raiz
@app.get("/")
async def root():
    return {"message": "Hello World222"}

# Endpoint para upload de imagem e previsão de emoção
@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Ler o arquivo de imagem e fazer a predição
        image = Image.open(BytesIO(await file.read()))

        # Verificar se a imagem é RGBA e converter para RGB
        emotion = prever_emocao(image)
        return {"emocao": emotion}
    except UnidentifiedImageError:
        return {"error": "O arquivo enviado não é uma imagem válida."}
    except Exception as e:
        return {"error": f"Ocorreu um erro: {str(e)}"}
