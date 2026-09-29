import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { AlertTriangle, Bot, ChevronDown, ChevronUp, Heart, Send, Stethoscope } from 'lucide-react';

const SESSION_KEY = 'medai-session-id';
const API_BASE_URL = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');

function makeId() {
  return crypto.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
}

function parseSseFrame(frame) {
  const lines = frame.split(/\r?\n/);
  const event = lines.find((line) => line.startsWith('event:'))?.slice(6).trim();
  const data = lines.filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trim()).join('\n');
  return event && data ? { event, data: JSON.parse(data) } : null;
}

function summaryFor(result = {}) {
  if (result.deferred) return 'Waiting for an earlier tool result';
  if (result.matched) {
    const names = result.matched
      .map((item) => item.name || item.phrase)
      .filter(Boolean)
      .join(', ');
    const detail = names ? ` (${names})` : '';
    return `${result.matched.length} symptom${result.matched.length === 1 ? '' : 's'} matched${detail}`;
  }
  if (result.candidates) {
    const names = result.candidates
      .map((item) => item.name)
      .filter(Boolean)
      .join(', ');
    const detail = names ? ` (${names})` : '';
    return `${result.candidates.length} candidate${result.candidates.length === 1 ? '' : 's'} ranked${detail}`;
  }
  if (result.specialties) return result.specialties.map((item) => item.name).join(', ') || 'No specialty found';
  if (result.name) return result.name;
  if (result.error) return result.error;
  return 'Completed';
}

function ToolTrace({ activity }) {
  const [open, setOpen] = useState(false);
  if (!activity.length) return null;
  return (
    <section className="mx-auto w-full max-w-[85%] rounded-xl border border-slate-200 bg-white/80 p-3 text-xs text-slate-600 shadow-sm">
      <button type="button" onClick={() => setOpen((value) => !value)} className="flex w-full items-center justify-between font-semibold text-slate-700">
        <span>Engine activity · {activity.length} step{activity.length === 1 ? '' : 's'}</span>
        {open ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
      </button>
      {open && <div className="mt-3 space-y-2">{activity.map((item) => <div key={item.id} className="rounded-lg bg-slate-50 p-2"><div className="font-mono font-semibold text-teal-700">{item.name}</div><div>{item.result ? summaryFor(item.result) : 'Running…'}</div></div>)}</div>}
    </section>
  );
}

function AlertCard({ alert }) {
  const emergency = alert.level === 'emergency';
  return <div role="alert" aria-live={emergency ? 'assertive' : 'polite'} className={`mx-auto flex w-full max-w-[85%] gap-3 rounded-2xl border p-4 text-sm shadow-sm ${emergency ? 'border-red-500 bg-red-100 text-red-900' : 'border-amber-400 bg-amber-50 text-amber-950'}`}><AlertTriangle className={`mt-0.5 h-5 w-5 shrink-0 ${emergency ? 'text-red-700' : 'text-amber-700'}`} /><div><strong>{emergency ? 'Emergency alert' : 'Urgent alert'}: </strong>{alert.message}</div></div>;
}

export default function App() {
  const [messages, setMessages] = useState([{ id: 'welcome', sender: 'bot', text: 'Welcome to MedAi Clinic! How can I help you today?', isWelcome: true }]);
  const [activity, setActivity] = useState([]);
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const chatEndRef = useRef(null);
  const inputRef = useRef(null);

  useLayoutEffect(() => {
    const textarea = inputRef.current;
    const resize = () => {
      textarea.style.height = 'auto';
      textarea.style.height = `${textarea.scrollHeight + 2}px`;
    };
    resize();
    let previousWidth = textarea.clientWidth;
    const observer = new ResizeObserver(() => {
      if (textarea.clientWidth !== previousWidth) {
        previousWidth = textarea.clientWidth;
        resize();
      }
    });
    observer.observe(textarea);
    return () => observer.disconnect();
  }, [input]);

  useEffect(() => {
    // Effects may return only a cleanup function; do not implicitly return the
    // value of scrollIntoView (or a Promise) from this callback.
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, activity, isSending]);
  const addMessage = (message) => setMessages((current) => [...current, { id: makeId(), ...message }]);

  const handleEvent = (event, payload) => {
    if (event === 'tool_call') setActivity((current) => [...current, { id: makeId(), name: payload.name, args: payload.args, result: null }]);
    else if (event === 'tool_result') setActivity((current) => {
      const index = current.findIndex((item) => item.name === payload.name && !item.result);
      return index < 0 ? current : current.map((item, itemIndex) => itemIndex === index ? { ...item, result: payload.result } : item);
    });
    else if (event === 'alert') addMessage({ sender: 'alert', alert: payload });
    else if (event === 'message') {
      setActivity([]);
      addMessage({ sender: 'bot', text: payload.text });
    }
    else if (event === 'error') addMessage({ sender: 'bot', text: payload.message || 'The assistant could not complete that request.', isError: true });
    else if (event === 'done' && payload.session_id) localStorage.setItem(SESSION_KEY, payload.session_id);
  };

  const handleSend = async (event) => {
    event.preventDefault();
    const message = input.trim();
    if (!message || isSending) return;
    addMessage({ sender: 'user', text: message });
    setInput('');
    setActivity([]);
    setIsSending(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, session_id: localStorage.getItem(SESSION_KEY) }),
      });
      if (!response.ok || !response.body) throw new Error(`Request failed (${response.status})`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      
      while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const frames = buffer.split(/\r?\n\r?\n/);
        buffer = frames.pop() || '';
        for (const frame of frames) {
          const parsed = parseSseFrame(frame);
          if (parsed) handleEvent(parsed.event, parsed.data);
        }
        if (done) break;
      }
    } catch (error) {
      addMessage({ sender: 'bot', text: `Connection error: ${error.message}. Make sure the FastAPI server is running.`, isError: true });
    } finally {
      setIsSending(false);
    }
  };

  return <div className="flex min-h-screen w-full items-center justify-center bg-gradient-to-tr from-[#34D355] via-[#6EE7B7] to-[#93C5FF] p-4 font-sans antialiased selection:bg-[#B2DFDB]">
      <div className="relative flex h-[85vh] w-full max-w-5xl flex-col overflow-hidden rounded-3xl border border-white/50 bg-[#E2E8F0] shadow-xl">
        <header className="flex items-center justify-between border-b border-gray-100/50 bg-white p-4">
          <div className="flex items-center gap-2">
            <div className="relative -ml-1 flex h-14 w-14 items-center justify-center rounded-full border border-emerald-200 bg-[#E8F5E9] shadow-lg ring-4 ring-white motion-safe:animate-heart-float">
              <Heart className="h-7 w-7 fill-emerald-400/30 text-emerald-500" />
            </div>
            <div>
              <h1 className="text-base font-bold tracking-tight text-slate-800">MedAi Clinic</h1>
              <p className="flex items-center gap-1 text-xs font-medium text-emerald-600"><span className="inline-block h-1.5 w-1.5 rounded-full bg-emerald-500" /> Online Health Guide</p>
            </div>
          </div>
        </header>
        <main className="flex-1 space-y-4 overflow-y-auto p-4">
          {messages.map((message) => {
            if (message.sender === 'alert')
                return <AlertCard key={message.id} alert={message.alert} />;
            const isBot = message.sender === 'bot';
            return <div key={message.id} className={`flex max-w-[85%] items-end gap-2.5 ${isBot ? 'mr-auto' : 'ml-auto flex-row-reverse'}`}>
                {isBot &&
                  <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-sky-100 bg-[#E1F5FE]">
                    <Bot className="h-4 w-4 text-sky-600" />
                  </div>
                }
                <div className={`rounded-2xl p-3.5 text-sm font-semibold shadow-sm ${isBot ? (message.isError ? 'bg-red-100 text-red-950' : 'rounded-bl-none border border-sky-200/40 bg-gradient-to-br from-[#E0F2FE] to-[#BAE6FD] text-sky-950') : 'rounded-br-none bg-gradient-to-br from-[#2DD4BF] to-[#0D9488] text-white'}`}>
                  <p className="whitespace-pre-line leading-relaxed">{message.text}</p>
                </div>
              </div>;
          })}
          
          <ToolTrace activity={activity} />
          {isSending && 
            <div className="flex items-center gap-2 text-sm font-medium text-slate-500">
              <Stethoscope className="h-4 w-4 animate-pulse" /> 
              Consulting the triage engine…
            </div>}
            <div ref={chatEndRef} />
        </main>
        <footer className="shrink-0 border-t border-gray-100/40 bg-white p-3">
          <form onSubmit={handleSend} className="relative flex items-center">
            <textarea ref={inputRef} rows={1} aria-label="Message" value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault();
                event.currentTarget.form.requestSubmit();
              }
            }} placeholder="Describe your symptoms..." disabled={isSending} className="max-h-[40vh] w-full resize-none overflow-y-auto rounded-3xl border border-slate-200 bg-white py-3 pl-4 pr-12 text-sm font-medium text-slate-700 placeholder-slate-400 shadow-inner transition-colors focus:border-[#00BFA5] focus:outline-none focus:ring-4 focus:ring-teal-50 disabled:cursor-not-allowed disabled:bg-slate-100" />
            <button type="submit" aria-label="Send message" disabled={!input.trim() || isSending} className="absolute bottom-1.5 right-1.5 flex items-center justify-center rounded-full bg-[#00BFA5] p-2 text-white transition-all hover:bg-[#00a892] disabled:opacity-40">
            <Send className="h-4 w-4" /></button>
          </form>
          <p className="mt-2 text-center text-[11px] text-slate-400">Portfolio demonstration only — not a substitute for professional medical assessment.</p>
        </footer>
      </div>
    </div>
  ;
}
