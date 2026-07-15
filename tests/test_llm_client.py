from unittest.mock import patch, MagicMock

def test_call_groq_returns_parsed_json():
    from llm_client import call_groq
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"category": "Resources", "tags": ["test"], "summary": "A test"}'
    
    with patch("llm_client.Groq") as MockGroq:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockGroq.return_value = mock_client
        
        result = call_groq("Classify this note", "Some content")
        assert result["category"] == "Resources"
        assert "tags" in result

def test_call_groq_retries_on_invalid_json():
    from llm_client import call_groq
    
    # First call returns invalid JSON, second returns valid
    mock_response_bad = MagicMock()
    mock_response_bad.choices = [MagicMock()]
    mock_response_bad.choices[0].message.content = "not json at all"
    
    mock_response_good = MagicMock()
    mock_response_good.choices = [MagicMock()]
    mock_response_good.choices[0].message.content = '{"category": "Areas", "tags": ["retry"], "summary": "Retried"}'
    
    with patch("llm_client.Groq") as MockGroq:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [mock_response_bad, mock_response_good]
        MockGroq.return_value = mock_client
        
        result = call_groq("Classify this", "Content", max_retries=2)
        assert result["category"] == "Areas"
