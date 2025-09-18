from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser

config = {"output_format": "markdown", "use_llm": False}
config_parser = ConfigParser(config)

converter = PdfConverter(
    config=config_parser.generate_config_dict(),
    artifact_dict=create_model_dict(),
    processor_list=config_parser.get_processors(),
    renderer=config_parser.get_renderer(),
    llm_service=None,
)

rendered = converter("/Users/arthurbrenno/Documents/Dev/Paragon/agentle/examples/curriculum.pdf")
print(rendered)
