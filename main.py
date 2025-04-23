from config.config import Config
from config.Explorer import API2DExplorer

def main() -> None:
    config = Config()
    explorer = API2DExplorer(config.base_url, config.default_bounds, config.seed)
    explorer.run()


if __name__ == "__main__":
    main()
