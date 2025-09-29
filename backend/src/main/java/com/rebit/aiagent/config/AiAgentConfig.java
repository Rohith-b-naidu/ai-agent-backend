package com.rebit.aiagent.config;

import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.jlama.JlamaChatModel;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class AiAgentConfig {

    @Bean
    public ChatLanguageModel chatLanguageModel() {
        return JlamaChatModel.builder()
                .modelName("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
                .build();
    }

}
