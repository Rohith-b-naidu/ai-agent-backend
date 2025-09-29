package com.rebit.aiagent.dto;

public record DomElement(
        String type,
        String text,
        String value,
        String placeholder,
        String ariaLabel,
        String selector
) {}