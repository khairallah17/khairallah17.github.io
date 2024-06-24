---
layout: post
title:  "Machine Learning 101 - A Newbie's Chronicle"
categories: ["AI","Machine learning"]
---

<!-- ## Definition
**Arthur Samuel (1959)**: Machine Learning is the field of study that gives the computer the ability to learn without being explicitly programmed -->

## Machine Learning ?

Tom Mitchell (1998) defined machine learning as follows:

> "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

### Let's break down this definition step by step.

 - **Experience (E)**
    This refers to the data or experiences the computer program is exposed to. Experience **E** could be past data, interactions, or any form of input the program can learn from.

    *Example*: In a spam email filter, experience **E** would be the collection of emails that have been labeled as "spam" or "not spam."

 - **Class of Tasks (T)**
    These are the specific tasks that the program is designed to perform. The tasks should be clearly defined so the program knows what it is supposed to accomplish.

    *Example*: For the spam email filter, the task **T** is to classify incoming emails as "spam" or "not spam."

 - **Performance Measure (P)**
    This is the metric used to evaluate how well the program is performing the tasks. The performance measure needs to be quantifiable so improvements can be tracked.

    *Example*: In the spam email filter example, the performance measure **P** could be the accuracy of correctly classifying emails or the precision and recall of the filter.

 - **Putting It All Together**

    - **Learning from Experience (E)**: The program uses the data it has been exposed to (experience **E**) to make better decisions over time.

    - **With Respect to Tasks (T)**: The learning is specifically focused on improving performance in the given tasks (**T**) it is designed to perform.

    - **Performance Measure (P)**: Improvement is gauged using a specific performance measure (**P**), which quantifies how well the tasks are being performed.

