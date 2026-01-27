import { useState, useRef, useEffect } from 'react';
import { queryKnowledgeBase, clearHistory } from '../services/api';

export default function ChatInterface({ onCitationClick }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const question = input.trim();
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: question }]);
    setLoading(true);

    try {
      const response = await queryKnowledgeBase(question);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: response.answer,
          citations: response.citations,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'error', content: err.message },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = async () => {
    await clearHistory();
    setMessages([]);
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <h2>Ask Questions</h2>
        {messages.length > 0 && (
          <button onClick={handleClear} className="clear-btn">
            Clear
          </button>
        )}
      </div>

      <div className="messages">
        {messages.length === 0 && (
          <div className="empty-state">
            <p>Ask questions about your indexed videos.</p>
            <p className="hint">Try: "What are the main topics discussed?"</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            {msg.role === 'user' && <div className="message-content">{msg.content}</div>}

            {msg.role === 'assistant' && (
              <>
                <div className="message-content">{msg.content}</div>
                {msg.citations && msg.citations.length > 0 && (
                  <div className="citations">
                    <span className="citations-label">Sources:</span>
                    {msg.citations.map((cite, j) => (
                      <button
                        key={j}
                        className="citation-link"
                        onClick={() => onCitationClick?.(cite)}
                        title={cite.text_snippet}
                      >
                        {cite.video_title} @ {cite.timestamp}
                      </button>
                    ))}
                  </div>
                )}
              </>
            )}

            {msg.role === 'error' && (
              <div className="message-content error">{msg.content}</div>
            )}
          </div>
        ))}

        {loading && (
          <div className="message assistant loading">
            <div className="typing-indicator">
              <span></span><span></span><span></span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="chat-input">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}
