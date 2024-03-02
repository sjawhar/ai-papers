## Task Description
### Background
A common research task is classifying documents, but it’s expensive to classify all of them. Ideally, one should be able to manually classify 20 examples then have an ML model classify the rest of the documents for them. That’s your challenge in this task.

### Dataset
The dataset is available [here](https://drive.google.com/drive/u/1/folders/1re_PhEZzIxe8rAnO1cRcwVSAzNZv8teP). The goal is to classify whether an ArXiv paper is AI-relevant or not. There’s a train, dev, and test set, each with 500 entries. The test set doesn’t have labels. We provide all these data points to help you evaluate your solution, but as described above the solution should work with only 20 labeled examples.

Each entry contains information about an ArXiv paper:
* label: Whether the paper is AI-relevant
* text: The paper title and abstract, joined by a period
* meta: Metadata about the paper

### Task
Given this dataset, the task is to perform as well on it as possible, given only 20 labeled examples.
1. Test how well GPT-2 performs on it when applied in a straightforward way (few-shot learning with examples in prompt).
2. Experiment with changes that may improve it (e.g. adjustments to the prompt, using GPT-2 as part of more complex schemes, other models and training methods).
    * Don’t fine-tune GPT-2 or another model on the full training set, since in practice you will only have 20 labeled data points.
3. Deliverables:
    * Share a writeup with your findings on:
        * How well was GPT-2 able to perform on this task?
        * What tweaks that you tried worked vs. didn’t work?
        * What would you recommend based on these results? 
            * What would be good next steps?
    * Classify the test set and share a jsonl with the classifications.
    * Share your code.

Example utility functions related to GPT-2, loading in the data and building the prompt are in [this skeleton Colab](https://colab.research.google.com/drive/1kQPmsN8aINb0OHlaatvVHkFoZ4a2WcyO). Feel free to modify these.

### Evaluation Criteria
We’ll evaluate your project using these criteria, in approximate order of importance:
1. Code quality (readability, extensibility)
1. Optimization process for improving performance
1. Communication regarding results and recommendations
1. Performance on the classification task

We recognize that candidates have varying levels of pre-existing knowledge on working with language models and will do our best to take that into account.