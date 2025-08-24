import React, { useState } from "react";
import "./Chatbot.css";

export default function Chatbot() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([
    { from: "bot", text: "Hi 👋 I'm your Loan Assistant! Ask me about loans, schemes, or eligibility." }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  async function sendMessage(e) {
    e.preventDefault();
    if (!input.trim()) return;

    const newMsg = { from: "user", text: input };
    setMessages(prev => [...prev, newMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("http://127.0.0.1:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: newMsg.text })
      });

      const data = await res.json();
      setMessages(prev => [...prev, { from: "bot", text: data.reply || "Sorry, I didn’t understand that." }]);
    } catch (err) {
      setMessages(prev => [...prev, { from: "bot", text: "⚠️ Error connecting to server" }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="chatbot-container">
      <button className="chat-toggle" onClick={() => setOpen(!open)}>
        {open ? "✖" : "💬"}
      </button>

      {open && (
        <div className="chatbox">
          <div className="chat-header">Loan Assistant</div>
          <div className="chat-messages">
            {messages.map((m, i) => (
              <div key={i} className={`chat-message ${m.from}`}>
                {m.text}
              </div>
            ))}
            {loading && <div className="chat-message bot">⏳ Thinking...</div>}
          </div>
          <form className="chat-input" onSubmit={sendMessage}>
            <input
              type="text"
              value={input}
              placeholder="Type your question..."
              onChange={e => setInput(e.target.value)}
            />
            <button type="submit">➤</button>
          </form>
        </div>
      )}
    </div>
  );
}
