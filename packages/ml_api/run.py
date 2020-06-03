from api.app import create_ml_app
from api.config import DevelopmentConfig, ProductionConfig

app = create_ml_app(
    config_object = ProductionConfig
)

if __name__ == '__main__':
    app.run()
