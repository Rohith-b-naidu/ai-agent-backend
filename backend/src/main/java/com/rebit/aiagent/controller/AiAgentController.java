package com.rebit.aiagent.controller;

import com.rebit.aiagent.dto.AiRequest;
import com.rebit.aiagent.dto.DomElement;
import com.rebit.aiagent.service.OnnxEmbeddingModel;
import com.rebit.aiagent.service.ScreenHelpAgent;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.retriever.EmbeddingStoreRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/agent")
public class AiAgentController {

    @Autowired
    private EmbeddingModel embeddingModel;

    @Autowired
    private ChatLanguageModel chatLanguageModel;

    @GetMapping("/ai")
    public String health() {
        return "UP";
    }

    @PostMapping("/ask")
    public String ask(@RequestBody AiRequest request) {
        try {
            EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

            // Convert DOM elements to valid Documents
            List<Document> documents = request.domContext().stream()
                    .map(elem -> {
                        StringBuilder sb = new StringBuilder();

                        if (elem.type() != null) sb.append("Element type: ").append(elem.type()).append(". ");
                        if (elem.text() != null && !elem.text().isBlank()) sb.append("Visible text: ").append(elem.text()).append(". ");
                        if (elem.value() != null && !elem.value().isBlank()) sb.append("Value: ").append(elem.value()).append(". ");
                        if (elem.placeholder() != null && !elem.placeholder().isBlank()) sb.append("Placeholder: ").append(elem.placeholder()).append(". ");
                        if (elem.ariaLabel() != null && !elem.ariaLabel().isBlank()) sb.append("Accessibility label: ").append(elem.ariaLabel()).append(". ");
                        if (elem.selector() != null && !elem.selector().isBlank()) sb.append("Selector: ").append(elem.selector()).append(". ");

                        String docText = sb.toString().trim();
                        return docText.isBlank() ? null : new Document(docText);
                    })
                    .filter(Objects::nonNull)
                    .collect(Collectors.toList());

            // Convert to TextSegments safely
            List<TextSegment> segments = documents.stream()
                    .map(Document::toTextSegment)
                    .filter(Objects::nonNull)
                    .collect(Collectors.toList());

            if (segments.isEmpty()) {
                return "No valid DOM context to embed.";
            }

            segments.forEach(seg -> System.out.println("Embedding segment: " + seg.text()));

//            embeddingStore.addAll(embeddingModel.embedAll(segments).content());

            List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

            for (int i = 0; i < embeddings.size(); i++) {
                if (embeddings.get(i) == null) {
                    System.out.println("❌ Null embedding at index " + i + " for text: " + segments.get(i).text());
                }
            }

            List<TextSegment> validSegments = new ArrayList<>();
            for (int i = 0; i < embeddings.size(); i++) {
                if (embeddings.get(i) != null) {
                    validSegments.add(segments.get(i));
                }
            }
            List<Embedding> embeddingList = embeddingModel.embedAll(segments).content();


            embeddingStore.addAll(embeddingList, validSegments);

            EmbeddingStoreRetriever retriever = EmbeddingStoreRetriever.from(embeddingStore, embeddingModel);

            ScreenHelpAgent agent = AiServices.builder(ScreenHelpAgent.class)
                    .chatLanguageModel(chatLanguageModel)
                    .retriever(retriever)
                    .build();

            return agent.answer(request.query());

        } catch (Exception e) {
            e.printStackTrace();
            return "Error: " + e.getMessage();
        }
    }
}