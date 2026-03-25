(() => {
  const canvas = document.getElementById("game-canvas");
  const stageWrap = document.querySelector(".stage-wrap");
  const statusEl = document.getElementById("status");
  const roundEl = document.getElementById("round");
  const botsEl = document.getElementById("bots");
  const rendererApi = window.GaicaGameRenderer;

  if (!canvas || !stageWrap || !statusEl || !roundEl || !botsEl || !rendererApi) {
    return;
  }

  const renderer = new rendererApi.GameRenderer({
    canvas,
    stageWrap,
    assetPaths: rendererApi.createDefaultAssetPaths(),
  });

  async function pollState() {
    try {
      const response = await fetch("/api/state", { cache: "no-store" });
      const state = await response.json();
      renderer.setState(state);

      const series = state.series || null;
      statusEl.textContent = `Статус: ${state.status}`;
      const baseRound = `Тик: ${state.tick} | Время: ${state.time_seconds.toFixed(2)} сек | Карта: ${state.level.identifier}`;
      if (series && series.total_rounds > 1) {
        const score = series.score || {};
        const roundNo = Number(series.round || 0);
        const totalRounds = Number(series.total_rounds || 1);
        const s1 = Number(score["1"] || 0);
        const s2 = Number(score["2"] || 0);
        roundEl.textContent = `${baseRound} | Серия: ${roundNo}/${totalRounds} | Счёт: ${s1}:${s2}`;
      } else {
        roundEl.textContent = baseRound;
      }

      const connected = (state.bots_connected || []).join(", ") || "нет";
      const roles = (state.color_roles || [])
        .map((role) => {
          const side = role.color === "red" ? "Красный" : "Зелёный";
          const botLabel = role.bot_name || (role.bot_id ? `bot#${role.bot_id}` : "ожидание");
          return `${side}: ${botLabel}`;
        })
        .join(" | ");
      botsEl.textContent = `Боты подключены: ${connected}${roles ? ` | ${roles}` : ""}`;

      if (state.result) {
        if (state.result.reason === "series_score") {
          const score = state.result.series_score || {};
          const s1 = Number(score["1"] || 0);
          const s2 = Number(score["2"] || 0);
          const winner = state.result.winner_id ? `Победил бот #${state.result.winner_id}` : "Ничья";
          statusEl.textContent = `Матч завершён: ${winner} | Итоговый счёт ${s1}:${s2}`;
        } else {
          const winner = state.result.winner_id ? `Победил бот #${state.result.winner_id}` : "Ничья";
          statusEl.textContent = `Раунд завершён: ${winner} (${state.result.reason})`;
        }
      }
    } catch (err) {
      statusEl.textContent = "Ошибка чтения состояния сервера";
    }
  }

  async function bootstrap() {
    try {
      await renderer.loadAssets();
    } catch (err) {
      statusEl.textContent = `Ошибка загрузки ассетов: ${err}`;
      console.error(err);
      return;
    }

    renderer.resize();
    window.addEventListener("resize", () => renderer.resize());
    setInterval(pollState, 33);
    pollState();
    renderer.start();
  }

  bootstrap();
})();
