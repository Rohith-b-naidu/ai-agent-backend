package com.rebit.aiagent.service;

import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;

@SystemMessage("""
You are a helpful UI assistant that answers questions based ONLY on the provided context of UI elements on the screen.
Do not use any external knowledge or make assumptions. If the answer is not in the context, say "I don't have enough context to answer that."

The context is a list of UI elements in the format:
- Element type: [type]. Visible text: [text]. Value: [value]. Placeholder: [placeholder]. Accessibility label: [ariaLabel]. Selector: [selector].

Answer concisely and directly.
""")
public interface ScreenHelpAgent {

    String answer(@UserMessage String userQuery);
}