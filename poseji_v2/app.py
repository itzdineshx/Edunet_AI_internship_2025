from modules import main_ui, config

def main():
    config.inject_custom_css()
    main_ui.main()

if __name__ == "__main__":
    main()
