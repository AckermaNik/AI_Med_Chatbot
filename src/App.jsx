import { useState, useRef, useEffect } from 'react';
import { Send, User, Heart, Bot, Sparkles } from 'lucide-react';

export default function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: 'bot',
      text: "Welcome to MedAi Clinic! How can I help you today?",
      isWelcome: true
    }
  ]);
  const [input, setInput] = useState('');
  const chatEndRef = useRef(null);

  // Auto-scrolls chat window when a new response arrives
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = {
      id: Date.now(),
      sender: 'user',
      text: input
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');

    // Simulated Bot Response
    setTimeout(() => {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          sender: 'bot',
          text: "I am consulting the digital medical scrolls... I'll have an answer for you soon!"
        }
      ]);
    }, 1000);
  };

  return (
    <div className="w-full min-h-screen bg-gradient-to-tr from-[#34D355] via-[#6EE7B7] to-[#93C5FF] flex items-center justify-center p-4 font-sans antialiased selection:bg-[#B2DFDB]">      {/* Central View Device Container */}
      <div className="w-full max-w-5xl h-[85vh] bg-[#E2E8F0] rounded-3xl border border-white/50 shadow-xl flex flex-col overflow-hidden relative">
        
        {/* Top Header Bar */}
        <header className="p-4 flex items-center justify-between border-b border-gray-100/50 bg-white">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-full bg-[#E8F5E9] flex items-center justify-center border border-emerald-100 animate-pulse">
              <Heart className="w-5 h-5 text-emerald-500 fill-emerald-400/30" />
            </div>
            <div>
              <h1 className="text-base font-bold text-slate-800 tracking-tight">MedAi Clinic</h1>
              <p className="text-xs text-emerald-600 font-medium flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 inline-block animate-bounce" /> Online Health Guide
              </p>
            </div>
          </div>

          {/* User Profile Icon */}
          <button className="w-10 h-10 rounded-xl bg-slate-200/80 hover:bg-slate-300/80 active:scale-95 transition-all flex items-center justify-center border border-slate-300/30 group">
            <User className="w-5 h-5 text-slate-600 group-hover:rotate-12 transition-transform" />
          </button>
        </header>

        {/* Chat Feed */}
        <main className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-thumb-slate-200">
          {messages.map((msg) => {
            const isBot = msg.sender === 'bot';
            return (
              <div 
                key={msg.id} 
                className={`flex gap-2.5 items-end max-w-[85%] ${isBot ? 'mr-auto' : 'ml-auto flex-row-reverse animate-[slideInRight_0.2s_ease-out]'}`}
              >
                {/* Bot Profile Avatar Container */}
                {isBot && (
                  <div className={`rounded-full bg-[#E1F5FE] border border-sky-100 flex items-center justify-center flex-shrink-0 shadow-sm transition-all
                    ${msg.isWelcome ? 'w-9 h-9 border-2 ring-4 ring-emerald-50/50' : 'w-7 h-7'}`}
                  >
                    <Bot className={`text-sky-600 ${msg.isWelcome ? 'w-5 h-5' : 'w-4 h-4'}`} />
                  </div>
                )}

                {/* Speech Bubble Markup */}
                {/* Speech Bubble Markup */}
                <div 
                  className={`p-3.5 rounded-2xl text-sm font-semibold shadow-sm transition-all hover:shadow-md duration-300
                    ${isBot 
                      ? msg.isWelcome 
                        ? 'bg-gradient-to-br from-[#A7F3D0] to-[#86EFAC] text-emerald-950 rounded-bl-none border border-emerald-300/40 animate-[slideInLeft_0.25s_ease-out]'
                        : 'bg-gradient-to-br from-[#E0F2FE] to-[#BAE6FD] text-sky-950 rounded-bl-none border border-sky-200/40 animate-[slideInLeft_0.2s_ease-out]' 
                      : 'bg-gradient-to-br from-[#2DD4BF] to-[#0D9488] text-white rounded-br-none shadow-md shadow-teal-500/10'}`}
                >
                  <p className="leading-relaxed whitespace-pre-line">{msg.text}</p>
                </div>
              </div>
            );
          })}
          <div ref={chatEndRef} />
        </main>

        {/* Form Input Area */}
        <footer className="p-3 bg-white border-t border-gray-100/40">
          <form onSubmit={handleSend} className="relative flex items-center">
            {/* Tiny decorative sparkle icon inside the input */}
            <Sparkles className="absolute left-4 w-4 h-4 text-teal-400/70 pointer-events-none" />
            
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a medical question..."
              className="w-full bg-white border border-slate-200 rounded-full py-3 pl-10 pr-12 text-sm font-medium text-slate-700 placeholder-slate-400 focus:outline-none focus:border-[#00BFA5] focus:ring-4 focus:ring-teal-50 transition-all shadow-inner"
            />
            <button
              type="submit"
              disabled={!input.trim()}
              className="absolute right-1.5 p-2 rounded-full bg-[#00BFA5] hover:bg-[#00a892] text-white disabled:opacity-40 disabled:hover:bg-[#00BFA5] transition-all active:scale-95 flex items-center justify-center"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
        </footer>
      </div>
    </div>
  );
}