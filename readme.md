# Taxonomy-Guided Reasoning for Requirements Classification: A Study in Aerospace Industry (TRClass)

## Summary of Artifact
Requirements classification is crucial for managing and retrieving software requirements, especially when organizing them into a domain-specific taxonomy. However, traditional supervised classifiers require costly labeling, and zero-shot classification with large language models (LLMs) often fails on large, hierarchical label spaces. This paper proposes an LLM-enhanced hierarchical requirements classification approach guided by a domain taxonomy. We construct the domain-specific requirements taxonomy through automated preprocessing and decomposition of requirements, merging of requirement trees, and iterative expert-LLM refinement. Leveraging this taxonomy, we develop a Taxonomy-guided Requirements Classification (TRClass) method that reasons stepwise through the taxonomy: at each level, an LLM selects among child classes with a confidence score, optionally exploring multiple branches.


## Sun Search Control System
The mission of the Sun Search Control System (SSCS) is to perform sun localization and orientation by measuring the spacecraft's current attitude using gyroscopes, sun sensors, and star trackers, and it is the control software that rotates the spacecraft's direction along the pitch and roll axes, enabling the sun sensors to detect the Sun and maintain a sun-pointing orientation.

In this excel file, we present the labeled requirement descriptions according to the taxonomy.

## Codes
This repository contains the codes for the *TRClass*.

### Baselines
- Flat-Sentence: a zero-shot approach that uses a pretrained sentence embedding model to represent the requirement and each leaf class description, then assigns the requirement to the top-$k$ classes. This baseline treats all classes as flat candidates and does not use an LLM.
- Hier-Sentence: another baseline uses the hierarchy: it finds the most similar class at Level 1, then within that branch finds the most similar at Level 2, etc., until a leaf is reached. Essentially, it mirrors our taxonomy traversal but uses static embeddings instead of an LLM. 
- TELEClass: the state-of-the-art weakly supervised approach which utilizes an LLM (GPT-based) to generate training data and to score classes. This makes it a fair comparison for our approach, which also does not require labeled data for training.

### Abalation 
- Choose-Only: provides the LLM solely with a list of all possible classes to choose from.
- TRClass-NoShot: disables the few-shot retrieval, the LLM only gets the taxonomy and the requirement.
- TRClass-NoTaxo: removes the taxonomy-guided stepwise process, instead prompting the LLM to directly output all applicable classes in three shots.