# Legal-Document-Information-Extraction

This project focuses on the use of NLP techniques for the Information Extraction task from Vietnamese Legal Document, this includes the Metadata Extraction and Relation Extraction subtasks.

The training dataset for the NER model used for the ME task is collected manually by extracting the first/ last sentences from about 20 documents and annotated by Label-Studio.

The training dataset for the RE model is taken from a Previous work about Joint Entity Relation Extraction for Vietnamese Legal Document from Nguyen et al (2023): https://github.com/mlalab/VNLegalText/tree/main

This is a pre-requisite step for a larger project involving the construction of Vietnamese Legal Document Knowledge Graph for optimizing the Context Retrieval

To run a Demo of this Project result, follow the steps below:
1. Clone the Git repository

``` 

git clone https://github.com/GinHikat/Legal-Document-Information-Extraction.git
```

2. Follow the link below to download the artifact folder, containing index-mapping dictionaries, model state_dict, PhoBERT-base snapshot and VnCoreNLP local system.
[https://drive.google.com/file/d/1GKs4zLSR-8tIqdyE2ISvGY3QKUdAnrjX/view?usp=sharing](https://drive.google.com/file/d/1OK5m_rMpbJdzAO5zzAK2w4vETLaeyx4W/view?usp=sharing)

Unzip the file and put the artifact folder inside the demo folder

``` 
Legal-Document-Information-Extraction -> demo -> artifact -> data files

```

3. Navigate to the demo folder of the repository

``` 

cd Legal-Document-Information-Extraction/demo
```

4. Ensure the environment has Java > 1.8.0 for VnCoreNLP and run to download other dependencies
``` 
pip install -r requirement.txt

```

5. Run the demo.py file and input the path of a sample document for the prediction, 4 sample documents of different document types have been included in the path demo/artifact/test_document. 

``` 
python demo.py

Input file path here: artifact/test_document/luat_thue_gtgt_2024.pdf

```