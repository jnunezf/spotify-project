from api.app import create_ml_app
from api.config import DevelopmentConfig

app = create_ml_app(
    config_object = DevelopmentConfig
)

if __name__ == '__main__':
    app.run()
