package com.rebit.aiagent.dto;

import java.util.List;

public record AiRequest(
        String query,
        List<DomElement> domContext
) {}