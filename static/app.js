const form = document.querySelector("#predict-form");
const input = document.querySelector("#text-input");
const count = document.querySelector("#count");
const button = document.querySelector("#submit-button");
const message = document.querySelector("#message");
const result = document.querySelector("#result");
const prediction = document.querySelector("#prediction");
const probabilities = document.querySelector("#probabilities");
const inputNote = document.querySelector("#input-note");
const keyboard = document.querySelector("#keyboard");

const keyboardRows = [
  "1234567890-=",
  "qwertyuiop[]",
  "asdfghjkl;'",
  "zxcvbnm,./",
];

keyboardRows.forEach((characters) => {
  const row = document.createElement("div");
  row.className = "keyboard-row";
  [...characters].forEach((character) => {
    const key = document.createElement("span");
    key.className = "key";
    key.dataset.key = character;
    key.textContent = character;
    row.append(key);
  });
  keyboard.append(row);
});

const visualKeys = [...keyboard.querySelectorAll(".key")];

function updateKeyboard(value) {
  const validCharacters = [...value.toLowerCase()].filter((character) =>
    keyboardRows.some((row) => row.includes(character)),
  );
  const pressed = new Set(validCharacters);
  const latest = validCharacters.at(-1);

  visualKeys.forEach((key) => {
    key.classList.toggle("pressed", pressed.has(key.dataset.key));
    key.classList.toggle("latest", key.dataset.key === latest);
  });
}

input.addEventListener("input", () => {
  count.textContent = `${input.value.length} / 500`;
  message.textContent = "";
  updateKeyboard(input.value);
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
