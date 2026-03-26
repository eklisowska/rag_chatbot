# RAG Configuration Comparison Results

## Executive Summary

This document presents a comparative analysis of two RAG system configurations. **Config A** failed to retrieve relevant information about the ColPali approach, which prompted an adjustment to **Config B** with larger chunks and more retrieved results. The configuration change led to a successful retrieval of the ColPali information and generally more comprehensive answers.

---

## Data to analize

Raw logs available [here](logs.txt)

## Configuration Comparison


| Parameter       | Config A         | Config B         | Change |
| --------------- | ---------------- | ---------------- | ------ |
| EMBEDDING_MODEL | all-MiniLM-L6-v2 | all-MiniLM-L6-v2 | —      |
| CHUNK_SIZE      | 1000             | 1500             | +50%   |
| CHUNK_OVERLAP   | 200              | 300              | +50%   |
| WHISPER_MODEL   | base             | base             | —      |
| TOP_K_RESULTS   | 5                | 10               | +100%  |


**Rationale for Config B:** After Config A failed to find any information about the ColPali approach (returning "no mention of ColPali"), the configuration was adjusted to use larger chunks with more overlap and retrieve twice as many results. This hypothesis aimed to capture more context and increase the likelihood of finding relevant information spread across document sections.

---

## Results by Question

### Question 1: What are the production 'Do's' for RAG?


| Metric           | Config A      | Config B      |
| ---------------- | ------------- | ------------- |
| Chunks Retrieved | 5             | 10            |
| Avg Distance     | 0.555         | 0.656         |
| Best Distance    | 0.491         | 0.532         |
| Context Size     | ~1,257 tokens | ~3,289 tokens |
| Generation Time  | 6,464 ms      | 6,086 ms      |
| Total Time       | 17,486 ms     | 14,705 ms     |


**Sources Comparison:**

- **Config A:** 4 unique sources (1st Part_Productized Enterprise RAG.mp4, Productized & Enterprise RAG.pdf, RAG Intro.pdf, Architecture & Design Patterns.pdf)
- **Config B:** 5 unique sources (added 2nd Part_Architecture & Design Patterns.mp4)

**Answer Quality Analysis:**


| Aspect             | Config A                      | Config B                                                      |
| ------------------ | ----------------------------- | ------------------------------------------------------------- |
| Structure          | Single list                   | Categorized sections                                          |
| Categories Covered | Mixed list of practices       | Management, Data Protection, Technical Execution              |
| Depth              | Good coverage of techniques   | More comprehensive with strategic considerations              |
| Unique Points      | Query transformations details | Cost balancing, compliance (GDPR), infrastructure maintenance |


**Conclusion:** Config B produced a more structured and comprehensive answer with better categorization, despite having a higher average distance (lower similarity). The additional context allowed for better organization and coverage of governance aspects.

---

### Question 2: What is the difference between standard retrieval and the ColPali approach?


| Metric           | Config A      | Config B      |
| ---------------- | ------------- | ------------- |
| Chunks Retrieved | 5             | 10            |
| Avg Distance     | 0.540         | 0.588         |
| Best Distance    | 0.495         | 0.507         |
| Context Size     | ~1,261 tokens | ~3,757 tokens |
| Generation Time  | 3,766 ms      | 10,300 ms     |
| Total Time       | 3,798 ms      | 17,663 ms     |


**Sources Comparison:**

- **Config A:** 5 sources (Architecture & Design Patterns.pdf, 2 part Databases for GenAI.mp4, 2nd Part_Architecture & Design Patterns.mp4, Productized & Enterprise RAG.pdf, 1 part. RAG Intro.mp4)
- **Config B:** 7 unique sources (added RAG Intro.pdf, Databases for GenAI.pdf, 1st Part_Productized Enterprise RAG.mp4)

**Answer Quality Analysis:**


| Aspect         | Config A                          | Config B                                           |
| -------------- | --------------------------------- | -------------------------------------------------- |
| ColPali Found  | ❌ No                              | ✅ Yes                                              |
| Answer Content | Only described standard retrieval | Explained both approaches with comparison          |
| Key Insights   | None about ColPali                | OCR vs ColPali methodology, 15% improvement metric |


**Config A Answer**

> Based on the provided context, there is no mention of the "ColPali approach."
>
> The context describes **standard retrieval** as a process where a query/prompt is passed through an embedding model to identify relevant chunks in a vector store, which are then passed to an LLM as context. It also mentions advanced retrieval techniques (such as vector store indices, hierarchical indices, HyDE, and context enrichment) and post-retrieval steps (like reranking and query transformations), but it does not contain information regarding ColPali.

**Config B Answer**

> Based on the provided context, the differences between the standard retrieval approach and the ColPali approach are:
>
> - **Methodology:** The standard approach for retrieving information from PDFs typically utilizes OCR (Optical Character Recognition) to describe the documents before storing the information. ColPali is presented as an alternative approach to this OCR-based method.
> - **Performance:** ColPali shows better results for information retrieval than the standard system, with an improvement of approximately 15%.
> - **Usage:** While ColPali shows "quite interesting results" and may be used more often in the future, the context notes that the OCR approach is still the one mostly used currently.

**Conclusion:** This question demonstrates the critical impact of configuration on answer quality. Config A completely missed the ColPali information, while Config B successfully retrieved and synthesized a meaningful comparison. The additional sources (particularly the PDF documents) contained the relevant ColPali information.

---

### Question 3: Why is hybrid search better than vector-only search?


| Metric           | Config A      | Config B      |
| ---------------- | ------------- | ------------- |
| Chunks Retrieved | 5             | 10            |
| Avg Distance     | 0.547         | 0.566         |
| Best Distance    | 0.515         | 0.471         |
| Context Size     | ~1,303 tokens | ~3,858 tokens |
| Generation Time  | 4,496 ms      | 24,234 ms     |
| Total Time       | 4,522 ms      | 24,250 ms     |


**Sources Comparison:**

- **Config A:** 3 unique sources (1 part. RAG Intro.mp4, 2 part Databases for GenAI.mp4, 2nd Part_Architecture & Design Patterns.mp4)
- **Config B:** 4 unique sources (added Databases for GenAI.pdf)

**Answer Quality Analysis:**


| Aspect              | Config A                   | Config B                                             |
| ------------------- | -------------------------- | ---------------------------------------------------- |
| Accuracy Metric     | "more than 10%"            | "10.5% (from 43.8 to 48.4 NDCG@3)"                   |
| Sources Cited       | General mention            | Specific: Microsoft Azure AI Search, Anthropic study |
| Additional Insights | Reranker improves to 37.2% | 67% reduction in retrieval failures                  |
| Depth               | Good explanation           | More precise with benchmarks                         |


**Conclusion:** Both configurations provided quality answers, but Config B offered more precise metrics and cited specific studies (Microsoft Azure AI Search, Anthropic). Config B achieved the best single-chunk distance (0.471) across all questions, indicating highly relevant retrieval for this topic.

---

## Key Insights & Conclusions

### Performance Trade-offs


| Aspect                    | Config A (Smaller/Fewer) | Config B (Larger/More) |
| ------------------------- | ------------------------ | ---------------------- |
| **Context Size**          | ~1,250-1,300 tokens      | ~3,300-3,850 tokens    |
| **Avg Retrieval Quality** | Better (lower distance)  | Slightly worse         |
| **Answer Completeness**   | Good but can miss info   | More comprehensive     |
| **Generation Time**       | Faster (3.7-6.4s)        | Variable (6-24s)       |
| **Source Diversity**      | 3-5 sources              | 4-7 sources            |


### Key Findings

1. **Larger context prevents information gaps:** Config A's failure to find ColPali information demonstrates that smaller chunks and fewer results can miss critical information, especially when concepts are mentioned briefly or in specific document sections.
2. **More isn't always semantically closer:** Config B generally had higher average distances (lower similarity scores), yet produced more comprehensive answers. This suggests that retrieving more chunks captures information that might not rank highly but is still relevant.
3. **Source diversity matters:** Config B consistently retrieved from more unique sources, which contributed to more well-rounded answers with multiple perspectives and specific citations.
4. **Trade-off: Time vs. Completeness:** Config B's larger context led to longer generation times (especially visible in Q3: 24s vs 4.5s), but the quality improvement often justified the cost.

