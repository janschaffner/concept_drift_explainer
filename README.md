# Masterthesis

The focus of this Master's thesis is to bridge the gap between the detection of concept drifts in event logs and their practical interpretation. My initial research question is:  
**"How can external and internal context dimensions be integrated into process mining sense-making?"**

Currently, the CV4CDD framework for concept drift detection outputs change points in the form of JSON files. However, these are often not meaningful enough for analysts, as they lack explanations of *what* actually changed and *why*. 

My approach is to enrich these change points with relevant contextual data—such as organizational charts or new regulations—to help make sense of why a process has changed at a particular point in time. To achieve this, I use a large language model (specifically GPT-4o), which links the JSON output and event logs with contextual company data. This includes internal documents like Word files, PowerPoint presentations, intranet articles, and knowledge bases such as Confluence.

For example, if the intranet announces a new CIO taking office, the language model can associate that information with a nearby change point in the event logs. This helps analysts understand that a leadership change might be the underlying cause of the concept drift. In essence, the LLM serves as a bridge between the raw change points from the framework and the company-specific context, providing well-founded explanations for observed process changes.