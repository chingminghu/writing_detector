const form = document.querySelector("#predict-form");
const input = document.querySelector("#text-input");
const count = document.querySelector("#count");
const button = document.querySelector("#submit-button");
const message = document.querySelector("#message");
const result = document.querySelector("#result");
const prediction = document.querySelector("#prediction");
const probabilities = document.querySelector("#probabilities");
const inputNote = document.querySelector("#input-note");

input.addEventListener("input", () => {
  count.textContent = `${input.value.length} / 500`;
  message.textContent = "";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  message.textContent = "";
  button.disabled = true;
  button.querySelector("span").textContent = "辨識中…";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: input.value }),
    });
    const data = await response.json();

    if (!response.ok) {
      const detail = Array.isArray(data.detail)
        ? data.detail.map((item) => item.msg).join("、")
        : data.detail;
      throw new Error(detail || "辨識失敗，請稍後再試。");
    }

    prediction.textContent = data.prediction;
    probabilities.classList.remove("placeholder-bars");
    probabilities.replaceChildren(
      ...data.probabilities.map((value, digit) => {
        const row = document.createElement("div");
        row.className = digit === data.prediction ? "probability active" : "probability";
        row.innerHTML = `
          <span class="digit">${digit}</span>
          <span class="track"><span class="bar" style="width:${Math.max(value * 100, 0.6)}%"></span></span>
          <span class="percent">${(value * 100).toFixed(1)}%</span>
        `;
        return row;
      }),
    );

    inputNote.textContent = data.ignored_characters.length
      ? `已忽略不支援的字元：${data.ignored_characters.join(" ")}`
      : `實際讀取：${data.accepted_text}`;
  } catch (error) {
    message.textContent = error instanceof Error ? error.message : "無法連接伺服器。";
  } finally {
    button.disabled = false;
    button.querySelector("span").textContent = "開始辨識";
  }
});
