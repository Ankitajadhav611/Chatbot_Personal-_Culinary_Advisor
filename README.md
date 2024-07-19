# 🌟 Discover the Recipe Generating Chatbot: Your Personal Culinary Advisor! 🍽️

As part of my journey in learning Large Language Models (LLM) and implementing cool applications, here's our base model. Wondering what we did and how? Let’s walk you through it! 🚶‍♂️🚶‍♀️

## 🤖 What is this bot?
This chatbot answers the age-old question: “What to cook?” 🍳

**Examples:**
- “Hey, I have some leftover rice. What can I do with it?”
- "I don't have eggs. What can I use as a substitute in baking?"
- "What's the best way to grill vegetables?"

The chatbot provides the best recipes from its vast knowledge and detailed cooking instructions. 📚👨‍🍳

## 🧠 How does it work?
We delved into the popular buzzword LLM, which is everywhere! 🌍 We used the concept of Retrieval Augmented Generation (RAG) to develop a domain-specific knowledge application. RAG is a popular technique used by large businesses to create specialized applications. Learn more about RAG here: [RAG Article](https://www.promptingguide.ai/research/rag) 📖

**Key Concepts:**
- **Embeddings**: Vectorizing information to create a vector space of related data. Learn more about embeddings and vector space here: [Medium Article on Embeddings](https://medium.com/@vladris/embeddings-and-vector-databases-732f9927b377).
- **ChromaDB**: We chose ChromaDB as our vector database. Here's a step-by-step guide to setting it up: [ChromaDB Tutorial](https://www.datacamp.com/tutorial/chromadb-tutorial-step-by-step-guide).
- **Dataset**: Sourced from Hugging Face, The Food Processor dataset is a compilation of recipes that includes allergy information, dietary preferences, and alternative ingredients. Check it out here: [The Food Processor Dataset](https://huggingface.co/datasets/Thefoodprocessor/recipe_new_with_features_full).

We utilized powerful fine-tuned models like ChatGPT and Mixtral, ultimately selecting Mixtral for execution. 💪

## 🎉 Now Comes the Fun Part!
Implementing and integrating all these resources was a fun and educational experience. But the real goal was to develop a sharable, interactive chatbot accessible to anyone with a link. 🌐

**How did we achieve that?**
Thanks to Hugging Face, which provides an interface to host ML models. Learn more about hosting model demos with Hugging Face Spaces: [Hosting Model Demos](https://medium.com/the-owl/hosting-model-demos-with-hugging-face-spaces-and-streamlit-ea0db5f2dd54).

Enjoy exploring the chatbot! 🍲🤩

---

Feel free to further customize this description to better fit the specifics of your project or any additional details you'd like to include.
