package com.rebit.aiagent.service;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.springframework.stereotype.Service;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
//import ai.microsoft.onnxruntime.*;

import java.io.InputStream;
import java.nio.LongBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import ai.onnxruntime.*;

@Service
public class OnnxEmbeddingModel implements EmbeddingModel {

    private OrtEnvironment environment;
    private OrtSession session;
    private HuggingFaceTokenizer tokenizer;

    @PostConstruct
    public void init() {
        try {
            environment = OrtEnvironment.getEnvironment();
            InputStream modelStream = getClass().getResourceAsStream("/model/model.onnx");
            byte[] modelBytes = modelStream.readAllBytes();
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            session = environment.createSession(modelBytes, options);

//            InputStream tokenizerStream = getClass().getResourceAsStream("/model/tokenizer.json");

            Path tokenizerPath = Paths.get(
                    Objects.requireNonNull(getClass().getResource("/model/tokenizer.json")).toURI()
            );
            tokenizer = HuggingFaceTokenizer.newInstance(tokenizerPath);
        } catch (Exception e) {
            throw new RuntimeException("Failed to load ONNX model or tokenizer", e);
        }
    }

    @PreDestroy
    public void close() {
        if (session != null) {
            try {
                session.close();
            } catch (OrtException e) {
                // Ignore
            }
        }
        if (environment != null) {
            environment.close();
        }
    }

    @Override
    public dev.langchain4j.model.output.Response<List<Embedding>> embedAll(List<TextSegment> textSegments) {
        List<Embedding> embeddings = new ArrayList<>();
        for (TextSegment segment : textSegments) {
            embeddings.add(embed(segment.text()).content());
        }
        return new dev.langchain4j.model.output.Response<>(embeddings);
    }



    public Response<Embedding> embed(String text) {
        try {
            // 1. Tokenize
//            Map<String, Object> encoding = (Map<String, Object>) tokenizer.encode(text);
//            long[] inputIds = (long[]) encoding.get("input_ids");
//            long[] attentionMask = (long[]) encoding.get("attention_mask");
//            long[] tokenTypeIds = new long[inputIds.length];
            Encoding encoding = tokenizer.encode(text);
            long[] inputIds = encoding.getIds();
            long[] attentionMask = encoding.getAttentionMask();
            long[] tokenTypeIds = new long[inputIds.length]; // usually zeros

           // usually zeros for MiniLM

            // 2. Create ONNX tensors
            OnnxTensor inputIdsTensor = OnnxTensor.createTensor(environment, LongBuffer.wrap(inputIds), new long[]{1, inputIds.length});
            OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(environment, LongBuffer.wrap(attentionMask), new long[]{1, attentionMask.length});
            OnnxTensor tokenTypeIdsTensor = OnnxTensor.createTensor(environment, LongBuffer.wrap(tokenTypeIds), new long[]{1, tokenTypeIds.length});

            Map<String, OnnxTensor> inputs = Map.of(
                    "input_ids", inputIdsTensor,
                    "attention_mask", attentionMaskTensor,
                    "token_type_ids", tokenTypeIdsTensor
            );

            // 3. Run model
            try (OrtSession.Result result = session.run(inputs)) {
                Optional<OnnxValue> optionalValue = result.get("last_hidden_state");
                if (optionalValue.isEmpty()) {
                    throw new RuntimeException("Model output 'last_hidden_state' is missing");
                }

                OnnxValue value = optionalValue.get();
                if (!(value instanceof OnnxTensor)) {
                    throw new RuntimeException("'last_hidden_state' is not a tensor");
                }

                OnnxTensor outputTensor = (OnnxTensor) value;

                // 4. Extract and reshape output
                float[] flatHiddenStates = outputTensor.getFloatBuffer().array();
                long[] shape = outputTensor.getInfo().getShape(); // [1, seq_len, hidden_dim]
                int seqLen = (int) shape[1];
                int hiddenDim = (int) shape[2];

                float[][] hiddenStates2D = new float[seqLen][hiddenDim];
                for (int i = 0; i < seqLen; i++) {
                    System.arraycopy(flatHiddenStates, i * hiddenDim, hiddenStates2D[i], 0, hiddenDim);
                }

                // 5. Mean pooling & normalize
                float[] embedding = meanPool(hiddenStates2D, attentionMask);
                normalizeL2(embedding);

                return new Response(Embedding.from(embedding));
            }

        } catch (Exception e) {
            throw new RuntimeException("Failed to embed text", e);
        }
    }

    private float[] meanPool(float[][] hiddenStates, long[] attentionMask) {
        int seqLen = hiddenStates.length;
        int dim = hiddenStates[0].length;
        float[] pooled = new float[dim];
        int maskSum = 0;

        for (int i = 0; i < seqLen; i++) {
            if (attentionMask[i] == 1) {
                for (int j = 0; j < dim; j++) {
                    pooled[j] += hiddenStates[i][j];
                }
                maskSum++;
            }
        }

        if (maskSum > 0) {
            for (int j = 0; j < dim; j++) {
                pooled[j] /= maskSum;
            }
        }
        return pooled;
    }

    private void normalizeL2(float[] vector) {
        float norm = 0;
        for (float v : vector) {
            norm += v * v;
        }
        norm = (float) Math.sqrt(norm);
        if (norm > 0) {
            for (int i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        }
    }
}