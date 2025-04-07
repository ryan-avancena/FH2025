from mediapipe.tasks import python
from mediapipe.tasks.python import text

# Path to your TFLite model
model_path = 'bert_classifier.tflite'

# Define the base and classifier options
base_options = python.BaseOptions(model_asset_path=model_path)
options = text.TextClassifierOptions(base_options=base_options)

# Define the input text
input_text = 'As a mindless and thoughtless nostalgia trip that brings Minecraft to ' \
'life for devoted fans of the game, A Minecraft Movie will likely be enough. ' \
'But it could have been so much more.'

# Create the classifier and classify the text
with text.TextClassifier.create_from_options(options) as classifier:
    classification_result = classifier.classify(input_text)

# Print the result
# print(classification_result)

for category in classification_result.classifications[0].categories:
    print(f"Label: {category.category_name}, Score: {category.score}")