import React, {useState} from "react";
import "./Chatbot.css";

export default function Chatbot(){
  const [msgs, setMsgs] = useState([{sender:"bot", text:"Hi! I'm EcoCred Bot. Ask about loans, schemes, or type 'predict' to run a manual check."}]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const push = m => setMsgs(prev=>[...prev,m]);

  async function send(){
    if(!input.trim()) return;
    push({sender:"user", text:input});
    setInput(""); setLoading(true);

    try {
      // If user starts with "predict:" then attempt to parse payload after colon as JSON (optional)
      if(input.toLowerCase().startsWith("predict:")){
        const after = input.slice(8).trim();
        let payload = {};
        try { payload = JSON.parse(after); }
        catch(e){ 
          push({sender:"bot", text:"To run a prediction from chat use: predict: {\"income\":60000, \"loan_amount\":20000, ...} (json)"}); 
          setLoading(false); return;
        }
        const res = await fetch("http://127.0.0.1:5000/predict", {
          method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)
        });
        const data = await res.json();
        push({sender:"bot", text: data.chatbot_message || JSON.stringify(data)});
        setLoading(false); return;
      }

      // otherwise, call /chat
      const res = await fetch("http://127.0.0.1:5000/chat", {
        method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({message:input})
      });
      const data = await res.json();
      push({sender:"bot", text: data.reply || data.response || "No reply"});
    } catch(err){
      console.error(err);
      push({sender:"bot", text:"Error: unable to reach server."});
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="chatbot-outer card">
      <h3>Chat Assistant</h3>
      <div className="chat-window" id="chatwindow" style={{height:380, overflowY:"auto"}}>
        {msgs.map((m,i)=>(
          <div key={i} className={`message ${m.sender}`}>
            {m.text}
          </div>
        ))}
        {loading && <div className="message bot">Bot is typing...</div>}
      </div>

      <div style={{display:"flex", marginTop:8}}>
        <input
          className="field"
          value={input}
          onChange={e=>setInput(e.target.value)}
          onKeyDown={e=>e.key==="Enter" && send()}
          placeholder="Ask: e.g. Which schemes can I apply for?"
        />
        <button className="btn" onClick={send} style={{marginLeft:8}}>Send</button>
      </div>
      <div className="small" style={{marginTop:8}}>
        Tip: type <code>predict: {"{...json...}"}</code> to run a quick prediction from chat.
      </div>
    </div>
  );
}
