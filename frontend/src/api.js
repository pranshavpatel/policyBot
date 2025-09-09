import axios from "axios";
const API_BASE = import.meta.env.VITE_API_BASE || ""; // use Vite proxy if empty

export async function askAgent(message, trace=false) {
  const url = API_BASE ? `${API_BASE}/agent` : `/agent`;
  try {
    const res = await axios.post(url, { message, trace });
    return res.data;
  } catch (err) {
    const detail = err?.response?.data || err?.message || String(err);
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
}
