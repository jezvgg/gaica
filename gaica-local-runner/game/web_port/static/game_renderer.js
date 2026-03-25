(function attachGaicaGameRenderer(global) {
  const TILE_SIZE = 64;
  const TILESET_COLS = 6;
  const HOTLIME_TILE_SIZE = 12;
  const PROP_COLS = 10;
  const LEG_FRAME_SIZE = 32;
  const CHARACTER_FRAME_SIZE = 64;
  const DOOR_FRAME_SIZE = 38;
  const DOOR_RENDER_LENGTH = 56;
  const HOTLIME_FLOOR_BASE_INDEX = 197;
  const HOTLIME_FLOOR_ALT_INDEX = 173;
  const HOTLIME_BG_BASE_INDICES = [144, 154, 0, 14];
  const HOTLIME_BG_DETAIL_INDICES = [56, 57, 58, 69, 70, 71, 82, 83, 84];
  const HOTLIME_WALL_HORIZONTAL_INDEX = 4;
  const HOTLIME_WALL_VERTICAL_INDEX = 131;
  const HOTLIME_LETTERBOX_INDEX = 22;
  const HOTLIME_BOX_INDEX = 23;
  const LEVEL_BACKGROUND_SCALE = 4;
  const LEVEL_BACKGROUND_OFFSET_AMOUNT = 5;
  const TRAFFIC_MARGIN = 180;
  const TRAFFIC_SKINS = ["carYellow", "carRed", "carGreen", "carBlue"];
  const LEVEL_BACKGROUND_ROAD_COLS = [22, 99, 181, 233, 305, 352];
  const LEVEL_BACKGROUND_ROAD_ROWS = [27, 66, 105, 121, 156, 251];
  const WALL_STRIP_THICKNESS = 6;
  const GLASS_STRIP_THICKNESS = 6;
  const DOOR_HINGE_HALF_LENGTH = TILE_SIZE / 2 - WALL_STRIP_THICKNESS;
  const SIMULATION_TICK_RATE = 30;
  const BOX_RENDER_OVERSCAN = 2;
  const HOTLIME_BOX_SOURCE_INSET = 24;
  const HOTLIME_BOX_SOURCE_SIZE = 16;

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function smoothstep(edge0, edge1, value) {
    const width = Math.max(1e-6, edge1 - edge0);
    const t = clamp((value - edge0) / width, 0, 1);
    return t * t * (3 - 2 * t);
  }

  function easeOutCubic(value) {
    const t = clamp(value, 0, 1);
    return 1 - ((1 - t) ** 3);
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });
  }

  function createDefaultAssetPaths(basePath = "/assets/art/hotlime") {
    const normalizedBase = String(basePath || "/assets/art/hotlime").replace(/\/+$/, "");
    return {
      tileset: `${normalizedBase}/tileset.png`,
      background: `${normalizedBase}/background.png`,
      levelBackground: `${normalizedBase}/level_background.png`,
      floor: `${normalizedBase}/floor.png`,
      walls: `${normalizedBase}/walls.png`,
      props: `${normalizedBase}/props.png`,
      door: `${normalizedBase}/door.png`,
      legs: `${normalizedBase}/legs.png`,
      guns: `${normalizedBase}/guns.png`,
      bullet: `${normalizedBase}/projectiles.png`,
      carYellow: `${normalizedBase}/car_yellow.png`,
      carRed: `${normalizedBase}/car_red.png`,
      carGreen: `${normalizedBase}/car_green.png`,
      carBlue: `${normalizedBase}/car_blue.png`,
      lemon: `${normalizedBase}/lemon.png`,
      lime: `${normalizedBase}/lime.png`,
      orange: `${normalizedBase}/orange.png`,
      grapefruit: `${normalizedBase}/grapefruit.png`,
    };
  }

  function createRenderer(options) {
    const canvas = options && options.canvas;
    if (!(canvas instanceof HTMLCanvasElement)) {
      throw new Error("GaicaGameRenderer requires a canvas element");
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("Failed to acquire 2D context for renderer");
    }
    ctx.imageSmoothingEnabled = false;

    const stageWrap = (options && options.stageWrap) || canvas.parentElement || canvas;
    const minWidth = Number((options && options.minWidth) || 640);
    const minHeight = Number((options && options.minHeight) || 360);
    const assetPaths = (options && (options.assetPaths || options.imagePaths)) || {};
    const imagePaths = { ...createDefaultAssetPaths(), ...assetPaths };

    const images = {};
    const hotlimeCellCache = new Map();
    const backgroundSceneCache = new Map();
    let latestState = null;
    let animationTime = 0;
    let lastStateTick = -1;
    let rafId = 0;

    const effectTimeline = new Map();
    const camera = {
      x: 0,
      y: 0,
      scale: 1,
      initialized: false,
    };
    const parallax = {
      x: 0,
      y: 0,
      initialized: false,
    };
    const doorMotion = new Map();

    function drawPixelDot(x, y, size, color, alpha = 1.0) {
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.fillStyle = color;
      const s = Math.max(1, Math.floor(size));
      ctx.fillRect(Math.round(x) - Math.floor(s / 2), Math.round(y) - Math.floor(s / 2), s, s);
      ctx.restore();
    }

    function drawPixelLine(sx, sy, ex, ey, step, size, color, alpha = 1.0) {
      const dx = ex - sx;
      const dy = ey - sy;
      const dist = Math.hypot(dx, dy);
      if (dist <= 1e-6) {
        drawPixelDot(sx, sy, size, color, alpha);
        return;
      }

      const count = Math.max(1, Math.floor(dist / Math.max(1, step)));
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.fillStyle = color;
      const pxSize = Math.max(1, Math.floor(size));
      for (let i = 0; i <= count; i += 1) {
        const t = i / count;
        const x = sx + dx * t;
        const y = sy + dy * t;
        ctx.fillRect(
          Math.round(x) - Math.floor(pxSize / 2),
          Math.round(y) - Math.floor(pxSize / 2),
          pxSize,
          pxSize,
        );
      }
      ctx.restore();
    }

    function syncEffectTimeline(state) {
      const tick = Number((state && state.tick) || 0);
      if (lastStateTick >= 0 && tick < lastStateTick) {
        effectTimeline.clear();
      }
      lastStateTick = tick;

      const nowMs = performance.now();
      const active = new Set();
      for (const effect of (state && state.effects) || []) {
        const id = Number(effect.id);
        if (!Number.isFinite(id)) continue;
        active.add(id);
        const durationSec = Math.max(0.001, Number(effect.duration || 0.2));
        const entry = effectTimeline.get(id);
        if (!entry || entry.type !== effect.type) {
          effectTimeline.set(id, { startMs: nowMs, durationSec, type: effect.type });
        } else {
          entry.durationSec = durationSec;
        }
      }

      for (const id of effectTimeline.keys()) {
        if (!active.has(id)) {
          effectTimeline.delete(id);
        }
      }
    }

    function effectProgress(effect, fallbackDurationSec) {
      const id = Number(effect.id);
      const fallbackDuration = Math.max(0.001, Number(fallbackDurationSec || 0.2));
      if (Number.isFinite(id)) {
        const entry = effectTimeline.get(id);
        if (entry) {
          const durationSec = Math.max(0.001, Number(entry.durationSec || fallbackDuration));
          return clamp((performance.now() - entry.startMs) / (durationSec * 1000), 0.0, 1.0);
        }
      }
      return 0.5;
    }

    async function loadAssets() {
      const entries = Object.entries(imagePaths);
      const loaded = await Promise.all(entries.map(([_, src]) => loadImage(src)));
      entries.forEach(([key], idx) => {
        images[key] = loaded[idx];
      });
    }

    function resize() {
      const rect = stageWrap.getBoundingClientRect();
      const width = Math.max(minWidth, Math.floor(rect.width || canvas.width || minWidth));
      const height = Math.max(minHeight, Math.floor(rect.height || canvas.height || minHeight));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        ctx.imageSmoothingEnabled = false;
        camera.initialized = false;
      }
    }

    function hasSemanticEnvironmentArt() {
      return Boolean(images.floor && images.walls && images.props && images.door);
    }

    function atlasRect(index, columns, frameSize) {
      return {
        sx: (index % columns) * frameSize,
        sy: Math.floor(index / columns) * frameSize,
        sw: frameSize,
        sh: frameSize,
      };
    }

    function drawAtlasRect(image, rect, dx, dy, dw, dh, alpha = 1.0) {
      if (!image || !rect) return;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.drawImage(image, rect.sx, rect.sy, rect.sw, rect.sh, dx, dy, dw, dh);
      ctx.restore();
    }

    function drawAtlasIndex(image, index, columns, frameSize, dx, dy, dw, dh, alpha = 1.0) {
      drawAtlasRect(image, atlasRect(index, columns, frameSize), dx, dy, dw, dh, alpha);
    }

    function atlasColumns(image, frameSize) {
      if (!image) return 1;
      return Math.max(1, Math.floor((Number(image.width) || frameSize) / frameSize));
    }

    function drawHotlimeAtlasIndex(image, index, dx, dy, dw, dh, alpha = 1.0) {
      drawAtlasIndex(image, index, atlasColumns(image, HOTLIME_TILE_SIZE), HOTLIME_TILE_SIZE, dx, dy, dw, dh, alpha);
    }

    function repeatOffset(value, period) {
      if (!Number.isFinite(period) || period <= 0) return 0;
      const raw = Number(value || 0) % period;
      return raw < 0 ? raw + period : raw;
    }

    function drawRepeatedAtlasRect(
      targetCtx,
      image,
      rect,
      dx,
      dy,
      dw,
      dh,
      alpha = 1.0,
      scaleX = 1.0,
      scaleY = 1.0,
      phaseX = 0,
      phaseY = 0,
    ) {
      if (!targetCtx || !image || !rect || dw <= 0 || dh <= 0) return;

      const tileW = Math.max(1, Math.round(rect.sw * scaleX));
      const tileH = Math.max(1, Math.round(rect.sh * scaleY));
      const offsetX = repeatOffset(phaseX, tileW);
      const offsetY = repeatOffset(phaseY, tileH);

      targetCtx.save();
      targetCtx.globalAlpha = alpha;
      for (let tileY = -offsetY; tileY < dh; tileY += tileH) {
        const visibleTop = Math.max(0, tileY);
        const visibleBottom = Math.min(dh, tileY + tileH);
        const visibleHeight = visibleBottom - visibleTop;
        if (visibleHeight <= 0) continue;

        const srcOffsetY = Math.max(0, visibleTop - tileY);
        const srcY = rect.sy + Math.floor((srcOffsetY / tileH) * rect.sh);
        const srcH = Math.max(1, Math.ceil((visibleHeight / tileH) * rect.sh));

        for (let tileX = -offsetX; tileX < dw; tileX += tileW) {
          const visibleLeft = Math.max(0, tileX);
          const visibleRight = Math.min(dw, tileX + tileW);
          const visibleWidth = visibleRight - visibleLeft;
          if (visibleWidth <= 0) continue;

          const srcOffsetX = Math.max(0, visibleLeft - tileX);
          const srcX = rect.sx + Math.floor((srcOffsetX / tileW) * rect.sw);
          const srcW = Math.max(1, Math.ceil((visibleWidth / tileW) * rect.sw));

          targetCtx.drawImage(
            image,
            srcX,
            srcY,
            srcW,
            srcH,
            dx + visibleLeft,
            dy + visibleTop,
            visibleWidth,
            visibleHeight,
          );
        }
      }
      targetCtx.restore();
    }

    function drawHotlimeRepeatedTile(image, index, dx, dy, dw, dh, alpha = 1.0, scaleX = 1.0, scaleY = 1.0, phaseX = dx, phaseY = dy) {
      drawRepeatedAtlasRect(
        ctx,
        image,
        atlasRect(index, atlasColumns(image, HOTLIME_TILE_SIZE), HOTLIME_TILE_SIZE),
        dx,
        dy,
        dw,
        dh,
        alpha,
        scaleX,
        scaleY,
        phaseX,
        phaseY,
      );
    }

    function worldCellHash(cellX, cellY) {
      const x = Math.trunc(cellX) + 0x9e3779b9;
      const y = Math.trunc(cellY) - 0x7f4a7c15;
      return (Math.imul(x, 0x85ebca6b) ^ Math.imul(y, 0xc2b2ae35)) >>> 0;
    }

    function hashString(value) {
      const text = String(value || "");
      let hash = 2166136261 >>> 0;
      for (let index = 0; index < text.length; index += 1) {
        hash ^= text.charCodeAt(index);
        hash = Math.imul(hash, 16777619) >>> 0;
      }
      return hash >>> 0;
    }

    function seededUnit(seed) {
      const x = Math.sin((Number(seed) + 1) * 12.9898 + 78.233) * 43758.5453123;
      return x - Math.floor(x);
    }

    function levelVisualKey(level) {
      return `${String((level && level.identifier) || "level")}|${Number((level && level.width) || 0)}|${Number((level && level.height) || 0)}`;
    }

    function levelBackdropRect(level) {
      const centerX = Number((level && level.width) || 0) * 0.5;
      const centerY = Number((level && level.height) || 0) * 0.5;
      const naturalWidth = images.levelBackground ? images.levelBackground.width * LEVEL_BACKGROUND_SCALE : 0;
      const naturalHeight = images.levelBackground ? images.levelBackground.height * LEVEL_BACKGROUND_SCALE : 0;
      const width = Math.max(naturalWidth, Number((level && level.width) || 0) + TILE_SIZE * 8);
      const height = Math.max(naturalHeight, Number((level && level.height) || 0) + TILE_SIZE * 8);
      return {
        centerX,
        centerY,
        x: centerX - width * 0.5,
        y: centerY - height * 0.5,
        width,
        height,
      };
    }

    function levelBackdropOffset(level) {
      const rect = levelBackdropRect(level);
      const targetX = clamp((camera.x - rect.centerX) * 0.035, -LEVEL_BACKGROUND_OFFSET_AMOUNT, LEVEL_BACKGROUND_OFFSET_AMOUNT);
      const targetY = clamp((camera.y - rect.centerY) * 0.035, -LEVEL_BACKGROUND_OFFSET_AMOUNT, LEVEL_BACKGROUND_OFFSET_AMOUNT);
      if (!parallax.initialized) {
        parallax.x = targetX;
        parallax.y = targetY;
        parallax.initialized = true;
      } else {
        parallax.x += (targetX - parallax.x) * 0.14;
        parallax.y += (targetY - parallax.y) * 0.14;
      }
      return { x: parallax.x, y: parallax.y };
    }

    function scaledBackdropOffset(level, factor = 1.0) {
      const offset = levelBackdropOffset(level);
      return {
        x: offset.x * factor,
        y: offset.y * factor,
      };
    }

    function backgroundScene(level) {
      const key = levelVisualKey(level);
      if (backgroundSceneCache.has(key)) {
        return backgroundSceneCache.get(key);
      }

      const rect = levelBackdropRect(level);
      const seed = hashString(key);
      const levelWidth = Number((level && level.width) || 0);
      const topCandidates = LEVEL_BACKGROUND_ROAD_ROWS
        .map((row) => rect.y + row * LEVEL_BACKGROUND_SCALE)
        .filter((value) => value < -40);
      const leftCandidates = LEVEL_BACKGROUND_ROAD_COLS
        .map((col) => rect.x + col * LEVEL_BACKGROUND_SCALE)
        .filter((value) => value < -40);
      const rightCandidates = LEVEL_BACKGROUND_ROAD_COLS
        .map((col) => rect.x + col * LEVEL_BACKGROUND_SCALE)
        .filter((value) => value > levelWidth + 40);
      const pick = (items, salt) => items[Math.floor(seededUnit(seed + salt) * items.length) % items.length];
      const laneSpecs = [
        ...(topCandidates.length ? [
          { axis: "x", direction: seededUnit(seed + 11) > 0.5 ? 1 : -1, coord: pick(topCandidates, 13) },
        ] : []),
        ...(leftCandidates.length ? [
          { axis: "y", direction: seededUnit(seed + 23) > 0.5 ? 1 : -1, coord: pick(leftCandidates, 29) },
        ] : []),
        ...(rightCandidates.length ? [
          { axis: "y", direction: seededUnit(seed + 31) > 0.5 ? 1 : -1, coord: pick(rightCandidates, 37) },
        ] : []),
      ];

      const cars = [];
      laneSpecs.forEach((lane, laneIndex) => {
        const seedOffset = laneIndex * 97 + 31;
        const variantIndex = Math.floor(seededUnit(seed + seedOffset * 17) * TRAFFIC_SKINS.length) % TRAFFIC_SKINS.length;
        cars.push({
          axis: lane.axis,
          direction: lane.direction,
          coord: lane.coord + (seededUnit(seed + seedOffset * 19) - 0.5) * 12,
          assetKey: TRAFFIC_SKINS[variantIndex],
          cyclePerSecond: 0.09 + seededUnit(seed + seedOffset * 23) * 0.04,
          phase: seededUnit(seed + seedOffset * 29),
          scale: 0.4 + seededUnit(seed + seedOffset * 31) * 0.08,
          alpha: 0.62 + seededUnit(seed + seedOffset * 37) * 0.08,
        });
      });

      const clouds = Array.from({ length: 7 }, (_, index) => ({
        anchorX: rect.x + rect.width * (0.08 + seededUnit(seed + index * 41) * 0.84),
        anchorY: rect.y + rect.height * (0.08 + seededUnit(seed + index * 43) * 0.84),
        radius: 64 + seededUnit(seed + index * 47) * 84,
        travel: 12 + seededUnit(seed + index * 53) * 20,
        speed: 0.015 + seededUnit(seed + index * 59) * 0.02,
        phase: seededUnit(seed + index * 61),
        alpha: 0.065 + seededUnit(seed + index * 67) * 0.05,
      }));

      const scene = { rect, cars, clouds };
      backgroundSceneCache.set(key, scene);
      return scene;
    }

    function hotlimeBackdropIndex(cellX, cellY) {
      const hash = worldCellHash(cellX, cellY);
      if ((hash % 11) < 3) {
        return HOTLIME_BG_DETAIL_INDICES[hash % HOTLIME_BG_DETAIL_INDICES.length];
      }
      return HOTLIME_BG_BASE_INDICES[hash % HOTLIME_BG_BASE_INDICES.length];
    }

    function characterJuicePalette(character) {
      const key = String(character || "").trim().toLowerCase();
      return {
        lemon: { juice: "#facc15", pulp: "#fff7b3", rind: "#ca8a04" },
        lime: { juice: "#84cc16", pulp: "#ecfccb", rind: "#4d7c0f" },
        orange: { juice: "#fb923c", pulp: "#ffedd5", rind: "#c2410c" },
        grapefruit: { juice: "#fb7185", pulp: "#fff1f2", rind: "#be185d" },
      }[key] || { juice: "#facc15", pulp: "#fff7b3", rind: "#ca8a04" };
    }

    function juiceStainGrowth(item, stateTime) {
      const createdAt = Number(item && item.created_at);
      const createdTick = Number(item && item.created_tick);
      const currentTick = Number(latestState && latestState.tick);
      const timeAge = Number.isFinite(createdAt) && Number.isFinite(stateTime)
        ? Math.max(0, stateTime - createdAt)
        : 0;
      const tickAge = Number.isFinite(createdTick) && Number.isFinite(currentTick)
        ? Math.max(0, (currentTick - createdTick) / SIMULATION_TICK_RATE)
        : 0;
      const age = Math.max(timeAge, tickAge);
      if (age <= 0 && !Number.isFinite(createdAt) && !Number.isFinite(createdTick)) {
        return { scale: 1.0, alpha: 1.0 };
      }

      const progress = smoothstep(0.0, 0.46, age);
      return {
        scale: 0.42 + easeOutCubic(progress) * 0.58,
        alpha: 0.34 + progress * 0.66,
      };
    }

    function tileCellOrigin(x, y) {
      return {
        x: Math.floor(Number(x || 0) / TILE_SIZE) * TILE_SIZE,
        y: Math.floor(Number(y || 0) / TILE_SIZE) * TILE_SIZE,
      };
    }

    function obstacleEdge(obstacle) {
      const center = obstacle && obstacle.center;
      const halfSize = obstacle && obstacle.half_size;
      if (!Array.isArray(center) || center.length < 2 || !Array.isArray(halfSize) || halfSize.length < 2) {
        return null;
      }

      const cx = Number(center[0] || 0);
      const cy = Number(center[1] || 0);
      const hx = Math.abs(Number(halfSize[0] || 0));
      const hy = Math.abs(Number(halfSize[1] || 0));
      const origin = tileCellOrigin(cx, cy);
      const localX = cx - origin.x;
      const localY = cy - origin.y;
      const vertical = hx <= hy;
      if (vertical) {
        return localX >= TILE_SIZE / 2 ? "right" : "left";
      }
      return localY >= TILE_SIZE / 2 ? "bottom" : "top";
    }

    function doorHingeEdge(obstacle) {
      const center = obstacle && obstacle.center;
      const halfSize = obstacle && obstacle.half_size;
      if (!Array.isArray(center) || center.length < 2 || !Array.isArray(halfSize) || halfSize.length < 2) {
        return null;
      }

      const cx = Number(center[0] || 0);
      const cy = Number(center[1] || 0);
      const hx = Math.abs(Number(halfSize[0] || 0));
      const hy = Math.abs(Number(halfSize[1] || 0));
      const origin = tileCellOrigin(cx, cy);
      const localX = cx - origin.x;
      const localY = cy - origin.y;
      const verticalDoor = hy >= hx;
      if (verticalDoor) {
        return localY >= TILE_SIZE / 2 ? "bottom" : "top";
      }
      return localX >= TILE_SIZE / 2 ? "right" : "left";
    }

    function drawTilesetIndex(index, dx, dy, dw, dh, alpha = 1.0) {
      if (!images.tileset) return;
      const sx = (index % TILESET_COLS) * TILE_SIZE;
      const sy = Math.floor(index / TILESET_COLS) * TILE_SIZE;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.drawImage(images.tileset, sx, sy, TILE_SIZE, TILE_SIZE, dx, dy, dw, dh);
      ctx.restore();
    }

    function drawLegacyTile(tile) {
      const size = tile.size || TILE_SIZE;
      if (!images.tileset) return;
      ctx.drawImage(
        images.tileset,
        tile.src_x,
        tile.src_y,
        TILE_SIZE,
        TILE_SIZE,
        tile.x,
        tile.y,
        size,
        size,
      );
    }

    function floorTileIndex(tile) {
      return Number(tile && tile.tile_id) === 8 ? HOTLIME_FLOOR_ALT_INDEX : HOTLIME_FLOOR_BASE_INDEX;
    }

    function drawSemanticFloorTile(tile) {
      const size = Number(tile && tile.size) || TILE_SIZE;
      if (!images.floor) {
        drawLegacyTile(tile);
        return;
      }

      const dx = Number(tile && tile.x) || 0;
      const dy = Number(tile && tile.y) || 0;
      drawHotlimeRepeatedTile(images.floor, floorTileIndex(tile), dx, dy, size, size, 1.0, 1.0, 1.0, dx, dy);
    }

    function drawTiles(level) {
      if (hasSemanticEnvironmentArt()) {
        for (const tile of level.floor_tiles || []) {
          drawSemanticFloorTile(tile);
        }
        return;
      }

      for (const tile of level.floor_tiles || []) drawLegacyTile(tile);
      for (const tile of level.top_tiles || []) drawLegacyTile(tile);
      for (const tile of level.small_tiles || []) drawLegacyTile(tile);
    }

    function drawLevelBackgroundSprite(level) {
      if (!images.levelBackground) return;

      const rect = levelBackdropRect(level);
      const offset = scaledBackdropOffset(level, 1.0);
      ctx.save();
      ctx.globalAlpha = 0.34;
      ctx.drawImage(
        images.levelBackground,
        rect.x + offset.x,
        rect.y + offset.y,
        images.levelBackground.width * LEVEL_BACKGROUND_SCALE,
        images.levelBackground.height * LEVEL_BACKGROUND_SCALE,
      );
      ctx.restore();
    }

    function drawBackdropTraffic(level) {
      const scene = backgroundScene(level);
      const rect = scene.rect;
      const offset = scaledBackdropOffset(level, 1.22);
      const travelX = rect.width + TRAFFIC_MARGIN * 2;
      const travelY = rect.height + TRAFFIC_MARGIN * 2;

      scene.cars.forEach((car) => {
        const image = images[car.assetKey] || images.carYellow;
        if (!image) return;

        const cycle = (animationTime * car.cyclePerSecond + car.phase) % 1;
        const baseProgress = cycle < 0 ? cycle + 1 : cycle;
        const progress = car.direction > 0 ? baseProgress : 1 - baseProgress;
        const x = car.axis === "x"
          ? rect.x - TRAFFIC_MARGIN + progress * travelX + offset.x
          : car.coord + offset.x;
        const y = car.axis === "y"
          ? rect.y - TRAFFIC_MARGIN + progress * travelY + offset.y
          : car.coord + offset.y;
        const rotation = car.axis === "x"
          ? (car.direction > 0 ? Math.PI * 1.5 : Math.PI * 0.5)
          : (car.direction > 0 ? Math.PI : 0);
        const drawWidth = image.width * car.scale;
        const drawHeight = image.height * car.scale;

        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(rotation);
        ctx.globalAlpha = 0.1 * car.alpha;
        ctx.fillStyle = "#020617";
        ctx.beginPath();
        ctx.ellipse(0, drawHeight * 0.12, drawWidth * 0.3, drawHeight * 0.18, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = car.alpha;
        ctx.drawImage(image, -drawWidth * 0.5, -drawHeight * 0.5, drawWidth, drawHeight);
        ctx.restore();
      });
    }

    function drawBackdropClouds(level) {
      const scene = backgroundScene(level);
      const offset = scaledBackdropOffset(level, 1.55);
      ctx.save();
      ctx.globalCompositeOperation = "screen";
      scene.clouds.forEach((cloud) => {
        const drift = Math.sin((animationTime * cloud.speed + cloud.phase) * Math.PI * 2) * cloud.travel;
        const x = cloud.anchorX + drift + offset.x;
        const y = cloud.anchorY + drift * 0.45 + offset.y;
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, cloud.radius);
        gradient.addColorStop(0, `rgba(248, 250, 252, ${cloud.alpha})`);
        gradient.addColorStop(0.55, `rgba(226, 232, 240, ${cloud.alpha * 0.52})`);
        gradient.addColorStop(1, "rgba(226, 232, 240, 0)");
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(x, y, cloud.radius, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.restore();
    }

    function drawBackdrop(level) {
      if (!images.background || images.levelBackground) {
        drawLevelBackgroundSprite(level);
        drawBackdropTraffic(level);
        drawBackdropClouds(level);
        return;
      }

      ctx.save();
      ctx.globalAlpha = 0.96;
      const offset = scaledBackdropOffset(level, 0.62);
      const marginCells = 3;
      const maxCellX = Math.max(0, Math.ceil((Number(level && level.width) || 0) / TILE_SIZE));
      const maxCellY = Math.max(0, Math.ceil((Number(level && level.height) || 0) / TILE_SIZE));
      for (let cellY = -marginCells; cellY < maxCellY + marginCells; cellY += 1) {
        for (let cellX = -marginCells; cellX < maxCellX + marginCells; cellX += 1) {
          drawHotlimeRepeatedTile(
            images.background,
            hotlimeBackdropIndex(cellX, cellY),
            cellX * TILE_SIZE + offset.x,
            cellY * TILE_SIZE + offset.y,
            TILE_SIZE,
            TILE_SIZE,
            1.0,
            1.0,
            1.0,
            cellX * TILE_SIZE,
            cellY * TILE_SIZE,
          );
        }
      }
      ctx.restore();
      drawLevelBackgroundSprite(level);
      drawBackdropTraffic(level);
      drawBackdropClouds(level);
    }

    function geometryKey(kind, center, halfSize) {
      if (!Array.isArray(center) || center.length < 2 || !Array.isArray(halfSize) || halfSize.length < 2) {
        return null;
      }
      const cx = Math.round(Number(center[0]) * 10) / 10;
      const cy = Math.round(Number(center[1]) * 10) / 10;
      const hx = Math.round(Number(halfSize[0]) * 10) / 10;
      const hy = Math.round(Number(halfSize[1]) * 10) / 10;
      return `${String(kind || "").toLowerCase()}|${cx}|${cy}|${hx}|${hy}`;
    }

    function breakableKind(variant) {
      const lowered = String(variant || "").trim().toLowerCase();
      if (lowered === "glass") return "glass";
      if (lowered === "box") return "box";
      return lowered;
    }

    function effectiveObstacles(state) {
      const breakablesByObstacleId = new Map();
      const breakablesByGeometry = new Map();
      for (const item of (state && state.breakables) || []) {
        const obstacleId = Number(item && (item.obstacle_id ?? item.obstacle_entity ?? item.id ?? item.entity));
        if (Number.isFinite(obstacleId) && obstacleId > 0) {
          breakablesByObstacleId.set(obstacleId, item);
        }
        const key = geometryKey(
          breakableKind(item && item.variant),
          item && (item.center || item.position),
          item && item.half_size,
        );
        if (key) {
          breakablesByGeometry.set(key, item);
        }
      }

      return ((state && state.obstacles) || []).map((obstacle) => {
        const obstacleId = Number(obstacle && (obstacle.id ?? obstacle.entity));
        let breakable = breakablesByObstacleId.get(obstacleId);
        if (!breakable) {
          const key = geometryKey(obstacle && obstacle.kind, obstacle && obstacle.center, obstacle && obstacle.half_size);
          if (key) {
            breakable = breakablesByGeometry.get(key);
          }
        }
        if (!breakable) return obstacle;

        const solid = obstacle.solid !== false && breakable.alive !== false;
        if (solid === obstacle.solid) return obstacle;
        return { ...obstacle, solid };
      });
    }

    function worldToScreen(x, y) {
      return {
        x: x * camera.scale + (canvas.width / 2 - camera.x * camera.scale),
        y: y * camera.scale + (canvas.height / 2 - camera.y * camera.scale),
      };
    }

    function drawMirroredRect(image, rect, dx, dy, dw, dh, flipX = false, flipY = false, alpha = 1.0) {
      if (!image || !rect) return;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.translate(dx + (flipX ? dw : 0), dy + (flipY ? dh : 0));
      ctx.scale(flipX ? -1 : 1, flipY ? -1 : 1);
      ctx.drawImage(image, rect.sx, rect.sy, rect.sw, rect.sh, 0, 0, dw, dh);
      ctx.restore();
    }

    function edgeSpriteRect(edge, origin, thickness, length = TILE_SIZE) {
      if (edge === "left") return { x: origin.x, y: origin.y, w: thickness, h: length };
      if (edge === "right") return { x: origin.x + TILE_SIZE - thickness, y: origin.y, w: thickness, h: length };
      if (edge === "top") return { x: origin.x, y: origin.y, w: length, h: thickness };
      return { x: origin.x, y: origin.y + TILE_SIZE - thickness, w: length, h: thickness };
    }

    function drawHotlimeWallEdge(obstacle) {
      const edge = obstacleEdge(obstacle);
      if (!edge) return;

      const [cx, cy] = obstacle.center;
      const origin = tileCellOrigin(cx, cy);
      const rect = edgeSpriteRect(edge, origin, WALL_STRIP_THICKNESS);
      const outerBorder = "#24162b";
      const wallBase = "#e7b49c";
      const wallWarmShadow = "#c8917a";
      const wallHighlight = "#f4d0bf";

      ctx.save();
      ctx.fillStyle = wallBase;
      ctx.fillRect(rect.x, rect.y, rect.w, rect.h);

      ctx.fillStyle = wallWarmShadow;
      if (edge === "left" || edge === "right") {
        ctx.fillRect(rect.x + 1, rect.y, Math.max(1, rect.w - 2), 1);
      } else {
        ctx.fillRect(rect.x, rect.y + 1, rect.w, Math.max(1, rect.h - 2));
      }

      ctx.fillStyle = outerBorder;
      if (edge === "left") {
        ctx.fillRect(rect.x, rect.y, 1, rect.h);
        ctx.fillRect(rect.x + rect.w - 1, rect.y, 1, rect.h);
      } else if (edge === "right") {
        ctx.fillRect(rect.x, rect.y, 1, rect.h);
        ctx.fillRect(rect.x + rect.w - 1, rect.y, 1, rect.h);
      } else if (edge === "top") {
        ctx.fillRect(rect.x, rect.y, rect.w, 1);
        ctx.fillRect(rect.x, rect.y + rect.h - 1, rect.w, 1);
      } else {
        ctx.fillRect(rect.x, rect.y, rect.w, 1);
        ctx.fillRect(rect.x, rect.y + rect.h - 1, rect.w, 1);
      }

      ctx.fillStyle = wallHighlight;
      if (edge === "left") {
        ctx.fillRect(rect.x + rect.w - 2, rect.y + 1, 1, Math.max(1, rect.h - 2));
      } else if (edge === "right") {
        ctx.fillRect(rect.x + 1, rect.y + 1, 1, Math.max(1, rect.h - 2));
      } else if (edge === "top") {
        ctx.fillRect(rect.x + 1, rect.y + rect.h - 2, Math.max(1, rect.w - 2), 1);
      } else {
        ctx.fillRect(rect.x + 1, rect.y + 1, Math.max(1, rect.w - 2), 1);
      }
      ctx.restore();
    }

    function glassVisualRect(obstacle) {
      const edge = obstacleEdge(obstacle);
      if (!edge) return null;
      const [cx, cy] = obstacle.center;
      const origin = tileCellOrigin(cx, cy);
      return {
        edge,
        ...edgeSpriteRect(edge, origin, GLASS_STRIP_THICKNESS),
      };
    }

    function drawHotlimeGlass(obstacle) {
      const rect = glassVisualRect(obstacle);
      if (!rect) return;

      ctx.save();
      if (obstacle.solid) {
        ctx.globalAlpha = 0.4;
        ctx.fillStyle = "#3f6fb9";
        ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
        ctx.globalAlpha = 0.68;
        ctx.strokeStyle = "#a7cbff";
        ctx.lineWidth = 1.1;
        ctx.strokeRect(rect.x + 0.5, rect.y + 0.5, rect.w - 1, rect.h - 1);
      } else {
        const seedBase = Number(obstacle.id || 0) * 97 + Math.floor(obstacle.center[0] * 0.73 + obstacle.center[1] * 1.19);
        const shardColorA = "#b7d6ff";
        const shardColorB = "#4d82d1";
        const count = Math.max(30, Math.floor((rect.w + rect.h) * 0.55));

        for (let i = 0; i < count; i += 1) {
          const seed = seedBase + i * 131;
          const r1 = (Math.sin(seed * 0.731) + 1) * 0.5;
          const r2 = (Math.sin((seed + 17) * 1.123) + 1) * 0.5;
          const r3 = (Math.sin((seed + 41) * 1.771) + 1) * 0.5;
          const r4 = (Math.sin((seed + 83) * 0.537) + 1) * 0.5;
          const r5 = (Math.sin((seed + 121) * 0.413) + 1) * 0.5;

          const side = Math.floor(r1 * 4) % 4;
          let px = rect.x;
          let py = rect.y;
          const offset = 1 + r2 * 3.5;
          const jitter = (r5 - 0.5) * 4.0;
          if (side === 0) {
            px = rect.x - offset;
            py = rect.y + r3 * rect.h + jitter;
          } else if (side === 1) {
            px = rect.x + rect.w + offset;
            py = rect.y + r3 * rect.h + jitter;
          } else if (side === 2) {
            px = rect.x + r3 * rect.w + jitter;
            py = rect.y - offset;
          } else {
            px = rect.x + r3 * rect.w + jitter;
            py = rect.y + rect.h + offset;
          }

          const shardW = 1 + Math.floor(r3 * 3);
          const shardH = 1 + Math.floor(r4 * 3);

          ctx.globalAlpha = 0.55 + r2 * 0.4;
          ctx.fillStyle = r1 > 0.52 ? shardColorA : shardColorB;
          ctx.fillRect(Math.round(px), Math.round(py), shardW, shardH);
        }
      }
      ctx.restore();
    }

    function drawHotlimeDoor(obstacle, openAmount) {
      const edge = doorHingeEdge(obstacle);
      if (!edge || !images.door) return;

      const [cx, cy] = obstacle.center;
      const frameIndex = {
        left: 0,
        right: 1,
        bottom: 2,
        top: 3,
      }[edge] ?? 0;
      const isHorizontalDoor = edge === "left" || edge === "right";
      const drawWidth = isHorizontalDoor ? DOOR_RENDER_LENGTH : DOOR_FRAME_SIZE;
      const drawHeight = isHorizontalDoor ? DOOR_FRAME_SIZE : DOOR_RENDER_LENGTH;
      const swing = openAmount * (Math.PI * 0.45);
      const pivotX = edge === "left" ? -DOOR_HINGE_HALF_LENGTH : edge === "right" ? DOOR_HINGE_HALF_LENGTH : 0;
      const pivotY = edge === "top" ? -DOOR_HINGE_HALF_LENGTH : edge === "bottom" ? DOOR_HINGE_HALF_LENGTH : 0;
      const spriteX = edge === "left"
        ? -DOOR_HINGE_HALF_LENGTH
        : edge === "right"
          ? DOOR_HINGE_HALF_LENGTH - drawWidth
          : -drawWidth * 0.5;
      const spriteY = edge === "top"
        ? -DOOR_HINGE_HALF_LENGTH
        : edge === "bottom"
          ? DOOR_HINGE_HALF_LENGTH - drawHeight
          : -drawHeight * 0.5;
      const direction = edge === "right" || edge === "bottom" ? -1 : 1;
      const alpha = 0.94 - openAmount * 0.28;

      ctx.save();
      ctx.translate(cx, cy);
      ctx.translate(pivotX, pivotY);
      ctx.rotate(direction * swing);
      ctx.translate(-pivotX, -pivotY);
      drawAtlasIndex(
        images.door,
        frameIndex,
        4,
        DOOR_FRAME_SIZE,
        spriteX,
        spriteY,
        drawWidth,
        drawHeight,
        alpha,
      );
      ctx.restore();
    }

    function drawHotlimeProp(index, cx, cy, size = TILE_SIZE, alpha = 1.0, overscan = 0) {
      const drawSize = size + overscan;
      drawAtlasIndex(
        images.props,
        index,
        PROP_COLS,
        TILE_SIZE,
        cx - drawSize * 0.5,
        cy - drawSize * 0.5,
        drawSize,
        drawSize,
        alpha,
      );
    }

    function drawHotlimeBox(cx, cy, w, h, alpha = 1.0) {
      const rect = atlasRect(HOTLIME_BOX_INDEX, PROP_COLS, TILE_SIZE);
      const drawW = w + BOX_RENDER_OVERSCAN;
      const drawH = h + BOX_RENDER_OVERSCAN;
      drawAtlasRect(
        images.props,
        {
          sx: rect.sx + HOTLIME_BOX_SOURCE_INSET,
          sy: rect.sy + HOTLIME_BOX_SOURCE_INSET,
          sw: HOTLIME_BOX_SOURCE_SIZE,
          sh: HOTLIME_BOX_SOURCE_SIZE,
        },
        cx - drawW * 0.5,
        cy - drawH * 0.5,
        drawW,
        drawH,
        alpha,
      );
    }

    function drawWorldObjects(state) {
      const obstacles = effectiveObstacles(state);
      const letterboxes = new Map((state.letterboxes || []).map((letterbox) => [letterbox.id, letterbox]));

      for (const obs of obstacles) {
        const [cx, cy] = obs.center;
        const [hx, hy] = obs.half_size;
        const x = cx - hx;
        const y = cy - hy;
        const w = hx * 2;
        const h = hy * 2;

        if (hasSemanticEnvironmentArt()) {
          if (obs.kind === "wall" && obs.solid) {
            drawHotlimeWallEdge(obs);
            continue;
          }

          if (obs.kind === "glass") {
            drawHotlimeGlass(obs);
            continue;
          }

          if (obs.kind === "door") {
            const prev = doorMotion.get(obs.id) || { open: obs.solid ? 0.0 : 1.0 };
            const target = obs.solid ? 0.0 : 1.0;
            prev.open += (target - prev.open) * 0.24;
            if (Math.abs(target - prev.open) < 0.01) prev.open = target;
            doorMotion.set(obs.id, prev);
            drawHotlimeDoor(obs, clamp(prev.open, 0.0, 1.0));
            continue;
          }

          if (obs.kind === "box" && obs.solid) {
            drawHotlimeBox(cx, cy, w, h);
            continue;
          }

          if (obs.kind === "letterbox") {
            const info = letterboxes.get(obs.id);
            const ready = info ? info.ready : false;
            const glow = ready ? 0.9 : 0.35;
            drawHotlimeProp(HOTLIME_LETTERBOX_INDEX, cx, cy);
            ctx.save();
            ctx.globalAlpha = glow * 0.3;
            ctx.fillStyle = "#facc15";
            ctx.fillRect(cx - 17, cy - 6, 34, 12);
            ctx.restore();
            continue;
          }
        }

        if (obs.kind === "box" && obs.solid) {
          drawTilesetIndex(5, cx - 32, cy - 32, 64, 64, 1.0);
          continue;
        }

        if (obs.kind === "glass") {
          ctx.save();
          if (obs.solid) {
            ctx.globalAlpha = 0.68;
            ctx.fillStyle = "#3f6fb9";
            ctx.fillRect(x, y, w, h);
            ctx.strokeStyle = "#a7cbff";
            ctx.lineWidth = 1.3;
            ctx.strokeRect(x, y, w, h);
            ctx.globalAlpha = 0.35;
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(x + w, y + h);
            ctx.moveTo(x + w, y);
            ctx.lineTo(x, y + h);
            ctx.stroke();
          } else {
            const seedBase = Number(obs.id || 0) * 97 + Math.floor(cx * 0.73 + cy * 1.19);
            const shardColorA = "#b7d6ff";
            const shardColorB = "#4d82d1";
            const count = Math.max(30, Math.floor((w + h) * 0.55));

            for (let i = 0; i < count; i += 1) {
              const seed = seedBase + i * 131;
              const r1 = (Math.sin(seed * 0.731) + 1) * 0.5;
              const r2 = (Math.sin((seed + 17) * 1.123) + 1) * 0.5;
              const r3 = (Math.sin((seed + 41) * 1.771) + 1) * 0.5;
              const r4 = (Math.sin((seed + 83) * 0.537) + 1) * 0.5;
              const r5 = (Math.sin((seed + 121) * 0.413) + 1) * 0.5;

              const side = Math.floor(r1 * 4) % 4;
              let px = x;
              let py = y;
              const offset = 1 + r2 * 3.5;
              const jitter = (r5 - 0.5) * 4.0;
              if (side === 0) {
                px = x - offset;
                py = y + r3 * h + jitter;
              } else if (side === 1) {
                px = x + w + offset;
                py = y + r3 * h + jitter;
              } else if (side === 2) {
                px = x + r3 * w + jitter;
                py = y - offset;
              } else {
                px = x + r3 * w + jitter;
                py = y + h + offset;
              }

              const shardW = 1 + Math.floor(r3 * 3);
              const shardH = 1 + Math.floor(r4 * 3);

              ctx.globalAlpha = 0.55 + r2 * 0.4;
              ctx.fillStyle = r1 > 0.52 ? shardColorA : shardColorB;
              ctx.fillRect(Math.round(px), Math.round(py), shardW, shardH);
            }

            ctx.globalAlpha = 0.82;
            ctx.fillStyle = "#f0feff";
            for (let i = 0; i < 10; i += 1) {
              const t = (i + 1) / 11;
              const side = i % 4;
              const distance = 3 + (i % 3);
              let sx = x;
              let sy = y;
              if (side === 0) {
                sx = x - distance;
                sy = y + t * h;
              } else if (side === 1) {
                sx = x + w + distance;
                sy = y + t * h;
              } else if (side === 2) {
                sx = x + t * w;
                sy = y - distance;
              } else {
                sx = x + t * w;
                sy = y + h + distance;
              }
              if (i % 2 === 0) {
                ctx.fillRect(Math.round(sx), Math.round(sy), 1, 4);
              } else {
                ctx.fillRect(Math.round(sx), Math.round(sy), 4, 1);
              }
            }
          }
          ctx.restore();
          continue;
        }

        if (obs.kind === "door") {
          const prev = doorMotion.get(obs.id) || { open: obs.solid ? 0.0 : 1.0 };
          const target = obs.solid ? 0.0 : 1.0;
          prev.open += (target - prev.open) * 0.24;
          if (Math.abs(target - prev.open) < 0.01) prev.open = target;
          doorMotion.set(obs.id, prev);

          const openAmount = clamp(prev.open, 0.0, 1.0);
          const vertical = h > w;
          const swing = openAmount * (Math.PI * 0.45);

          ctx.save();
          ctx.translate(cx, cy);
          ctx.globalAlpha = 0.92 - openAmount * 0.45;
          if (vertical) {
            ctx.translate(-w * 0.5, 0);
            ctx.rotate(-swing);
            ctx.translate(w * 0.5, 0);
          } else {
            ctx.translate(0, -h * 0.5);
            ctx.rotate(swing);
            ctx.translate(0, h * 0.5);
          }

          const grd = ctx.createLinearGradient(-w * 0.5, -h * 0.5, w * 0.5, h * 0.5);
          grd.addColorStop(0, "#b0895d");
          grd.addColorStop(1, "#6d4f2f");
          ctx.fillStyle = grd;
          ctx.fillRect(-w * 0.5, -h * 0.5, w, h);

          ctx.strokeStyle = "#4b3520";
          ctx.lineWidth = 1.0;
          ctx.strokeRect(-w * 0.5, -h * 0.5, w, h);

          ctx.globalAlpha = 0.35;
          ctx.fillStyle = "#22150a";
          if (vertical) {
            ctx.fillRect(-w * 0.5, -h * 0.45, w * 0.22, h * 0.9);
          } else {
            ctx.fillRect(-w * 0.45, -h * 0.5, w * 0.9, h * 0.22);
          }
          ctx.restore();
          continue;
        }

        if (obs.kind === "letterbox") {
          const info = letterboxes.get(obs.id);
          const ready = info ? info.ready : false;
          const glow = ready ? 0.9 : 0.35;
          drawTilesetIndex(6, x, y - 4, w, h * 2.0, 1.0);
          ctx.save();
          ctx.globalAlpha = glow * 0.35;
          ctx.fillStyle = "#facc15";
          ctx.fillRect(x - 1, y - 1, w + 2, h + 2);
          ctx.restore();
          continue;
        }
      }

      for (const [id, motion] of doorMotion.entries()) {
        if (Math.abs(motion.open) < 0.001) continue;
        const exists = obstacles.some((obs) => obs.id === id && obs.kind === "door");
        if (!exists) doorMotion.delete(id);
      }

      if (hasSemanticEnvironmentArt()) {
        return;
      }

      ctx.save();
      ctx.globalAlpha = 0.08;
      ctx.fillStyle = "#f1f5f9";
      for (const obs of obstacles) {
        if (obs.kind !== "wall" || !obs.solid) continue;
        const [cx, cy] = obs.center;
        const [hx, hy] = obs.half_size;
        ctx.fillRect(cx - hx, cy - hy, hx * 2, hy * 2);
      }
      ctx.restore();
    }

    function drawDebris(state) {
      const debris = state.debris || [];
      const stateTime = Number(state && state.time_seconds);
      for (const item of debris) {
        if (item.type === "box_debris") {
          const [x, y] = item.position;
          const size = Math.max(1, Number(item.size || 1));
          const tone = item.tone === "dark" ? "#7c5b3b" : "#b69263";
          drawPixelDot(x, y, size + 1, tone, 0.95);
          drawPixelDot(x, y, Math.max(1, size - 1), "#d6b98a", 0.72);
          continue;
        }

        if (item.type === "juice_stain") {
          const [x, y] = item.position;
          const palette = characterJuicePalette(item.character);
          const growth = juiceStainGrowth(item, stateTime);
          const radius = Math.max(10, Number(item.radius || 17) * 0.84 * growth.scale);
          const seed = Number(item.seed || 0);
          const leftShift = ((Math.sin(seed * 0.0017) + 1) * 0.5 - 0.5) * 2.4;
          const rightShift = ((Math.sin((seed + 17) * 0.0021) + 1) * 0.5 - 0.5) * 2.8;

          ctx.save();
          ctx.globalAlpha = 0.42 * growth.alpha;
          ctx.fillStyle = palette.rind;
          ctx.beginPath();
          ctx.arc(x, y + 1.8, radius + 1.2, 0, Math.PI * 2);
          ctx.fill();

          ctx.globalAlpha = 0.82 * growth.alpha;
          ctx.fillStyle = palette.juice;
          ctx.beginPath();
          ctx.arc(x, y + 1.4, radius - 1.6, 0, Math.PI * 2);
          ctx.fill();

          ctx.globalAlpha = 0.66 * growth.alpha;
          ctx.beginPath();
          ctx.arc(x - radius * 0.7 + leftShift, y + radius * 0.28, Math.max(4, radius * 0.38), 0, Math.PI * 2);
          ctx.fill();
          ctx.beginPath();
          ctx.arc(x + radius * 0.72 + rightShift, y + radius * 0.32, Math.max(4, radius * 0.42), 0, Math.PI * 2);
          ctx.fill();

          ctx.globalAlpha = 0.48 * growth.alpha;
          ctx.fillStyle = palette.pulp;
          ctx.beginPath();
          ctx.arc(x - 1.0, y + 0.8, Math.max(3, radius * 0.34), 0, Math.PI * 2);
          ctx.fill();

          for (let i = 0; i < 14; i += 1) {
            const angle = seed * 0.001 + (i / 14) * Math.PI * 2;
            const dist = radius * (0.45 + ((Math.sin((seed + i * 19) * 0.0027) + 1) * 0.5) * 0.55);
            const px = x + Math.cos(angle) * dist;
            const py = y + Math.sin(angle) * dist * 0.9 + 0.8;
            const size = i % 3 === 0 ? 2 : 1;
            drawPixelDot(px, py, size, palette.pulp, 0.78 * growth.alpha);
          }
          ctx.restore();
        }
      }
    }

    function drawPickup(pickup) {
      const isRevolver = pickup.type === "Revolver";
      const idx = isRevolver ? 1 : 4;
      const sx = (idx % 3) * CHARACTER_FRAME_SIZE;
      const sy = Math.floor(idx / 3) * CHARACTER_FRAME_SIZE;
      const x = pickup.position[0];
      const y = pickup.position[1];
      const ammo = Number.isFinite(pickup.ammo) ? pickup.ammo : 0;
      const isEmpty = ammo <= 0;
      const despawnRemaining = Number.isFinite(pickup.despawn_remaining) ? Math.max(0, pickup.despawn_remaining) : null;
      ctx.save();
      ctx.translate(x, y);
      if (isEmpty) {
        ctx.globalAlpha = 0.58;
      }
      ctx.drawImage(images.guns, sx, sy, CHARACTER_FRAME_SIZE, CHARACTER_FRAME_SIZE, -24, -24, 48, 48);
      if (isEmpty) {
        const badgeText = despawnRemaining !== null ? `${despawnRemaining.toFixed(1)}s` : "0";
        ctx.globalAlpha = 1.0;
        ctx.fillStyle = "rgba(15, 23, 42, 0.88)";
        ctx.fillRect(-18, 14, 36, 12);
        ctx.strokeStyle = "rgba(248, 113, 113, 0.95)";
        ctx.lineWidth = 1.2;
        ctx.strokeRect(-18, 14, 36, 12);
        ctx.fillStyle = "#fecaca";
        ctx.font = "bold 9px monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(badgeText, 0, 20);
      }
      ctx.restore();
    }

    function drawProjectile(projectile) {
      const x = projectile.position[0];
      const y = projectile.position[1];
      const velocity = projectile.velocity || [1.0, 0.0];
      const vx = velocity[0];
      const vy = velocity[1];
      const mag = Math.hypot(vx, vy) || 1.0;
      const nx = vx / mag;
      const ny = vy / mag;

      drawPixelDot(x, y, 3, "#fff8c2", 0.95);
      drawPixelDot(x, y, 2, "#ffe08a", 1.0);
      for (let i = 1; i <= 3; i += 1) {
        const tx = x - nx * (i * 2.3);
        const ty = y - ny * (i * 2.3);
        drawPixelDot(tx, ty, i === 1 ? 2 : 1, "#fcd34d", 0.78 - i * 0.15);
      }
    }

    function drawEffects(effects, mode = "all") {
      for (const effect of effects || []) {
        if (mode === "exclude_blood" && effect.type === "blood") {
          continue;
        }
        if (mode === "only_blood" && effect.type !== "blood") {
          continue;
        }

        if (effect.type === "tracer") {
          const [sx, sy] = effect.start;
          const [ex, ey] = effect.end;
          const outer = effect.weapon === "Uzi" ? "#ef4444" : "#f97316";
          const core = effect.weapon === "Uzi" ? "#ffe4d6" : "#fff0dc";
          drawPixelLine(sx, sy, ex, ey, 3.5, 3, outer, 0.5);
          drawPixelLine(sx, sy, ex, ey, 5.0, 2, core, 0.95);
          continue;
        }

        if (effect.type === "muzzle") {
          const [x, y] = effect.position;
          const warm = effect.weapon === "Uzi" ? "#f59e0b" : "#fbbf24";
          const hot = effect.weapon === "Uzi" ? "#fef08a" : "#fff7cc";
          drawPixelDot(x, y, 7, warm, 0.6);
          drawPixelDot(x, y, 4, hot, 0.95);
          for (let i = 0; i < 6; i += 1) {
            const angle = (i / 6) * Math.PI * 2 + (Number(effect.id || 0) % 3) * 0.2;
            drawPixelDot(x + Math.cos(angle) * 6, y + Math.sin(angle) * 6, 2, warm, 0.7);
          }
          continue;
        }

        if (effect.type === "impact") {
          const [x, y] = effect.position;
          const color = {
            wall: "#e2e8f0",
            glass: "#3f6fb9",
            box: "#d6b98a",
            door: "#c08457",
            player: "#fecaca",
          }[effect.material] || "#f1f5f9";
          drawPixelDot(x, y, 5, color, 0.85);
          drawPixelDot(x, y, 2, "#0f172a", 0.35);
          for (let i = 0; i < 5; i += 1) {
            const angle = (i / 5) * Math.PI * 2 + animationTime * 0.35;
            drawPixelDot(x + Math.cos(angle) * 5, y + Math.sin(angle) * 5, 2, color, 0.8);
          }
          continue;
        }

        if (effect.type === "blood") {
          const [x, y] = effect.position;
          const palette = characterJuicePalette(effect.character);
          const duration = Math.max(0.001, Number(effect.duration || 0.78));
          const progress = effectProgress(effect, duration);
          const settle = clamp(progress / 0.28, 0.0, 1.0);
          const fade = clamp((1.0 - progress) / 0.12, 0.32, 1.0);
          const baseSeed = Number(effect.id || 0) * 0.613;
          ctx.save();

          ctx.globalAlpha = 0.42 * fade;
          ctx.fillStyle = palette.rind;
          ctx.beginPath();
          ctx.arc(x + 0.6, y + 2.0, 16.8 + settle * 1.2, 0, Math.PI * 2);
          ctx.fill();

          ctx.globalAlpha = 0.82 * fade;
          ctx.fillStyle = palette.juice;
          ctx.beginPath();
          ctx.arc(x - 0.3, y + 1.6, 14.4 + settle * 1.5, 0, Math.PI * 2);
          ctx.fill();

          ctx.globalAlpha = 0.64 * fade;
          ctx.beginPath();
          ctx.arc(x - 11.5, y + 4.6, 6.4 + settle * 0.6, 0, Math.PI * 2);
          ctx.fill();
          ctx.beginPath();
          ctx.arc(x + 11.2, y + 4.8, 7.0 + settle * 0.7, 0, Math.PI * 2);
          ctx.fill();

          ctx.globalAlpha = 0.44 * fade;
          ctx.fillStyle = palette.pulp;
          ctx.beginPath();
          ctx.arc(x - 1.0, y + 1.0, 6.2 + settle * 0.5, 0, Math.PI * 2);
          ctx.fill();

          for (let i = 0; i < 16; i += 1) {
            const seed = baseSeed + i * 1.721;
            const r1 = (Math.sin(seed * 0.917) + 1) * 0.5;
            const r2 = (Math.sin((seed + 19) * 1.103) + 1) * 0.5;
            const r3 = (Math.sin((seed + 41) * 0.683) + 1) * 0.5;
            const angle = baseSeed + (i / 16) * Math.PI * 2;
            const distance = 6.0 + r1 * (13.5 + settle * 3.8);
            const px = x + Math.cos(angle) * distance * (1.05 + settle * 0.1);
            const py = y + Math.sin(angle) * distance * 0.68 + 1.4 + (r2 - 0.5) * 2.5;
            const size = r3 > 0.54 ? 2 : 1;
            drawPixelDot(px, py, size, palette.pulp, (0.7 + r2 * 0.24) * fade);
          }

          ctx.restore();
          continue;
        }

        if (effect.type === "break") {
          const [x, y] = effect.position;
          const base = Number(effect.id || 0) * 0.731;
          const shardCount = effect.variant === "Glass" ? 12 : 9;
          ctx.save();
          ctx.globalAlpha = 0.8;
          for (let i = 0; i < shardCount; i += 1) {
            const n = i + 1;
            const angle = base + (n / shardCount) * Math.PI * 2 + animationTime * 1.1;
            const radius = 6 + ((n * 7) % 5) * 2.3;
            const sx = x + Math.cos(angle) * radius;
            const sy = y + Math.sin(angle) * radius;
            if (effect.variant === "Glass") {
              drawPixelLine(sx, sy, sx + Math.cos(angle) * 3.2, sy + Math.sin(angle) * 3.2, 1.5, 2, "#76a7ea", 0.88);
              drawPixelDot(sx, sy, 2, "#c9deff", 0.9);
            } else {
              ctx.fillStyle = "#d6b98a";
              ctx.fillRect(sx - 1.5, sy - 1.5, 3, 3);
            }
          }
          ctx.restore();
          continue;
        }

        if (effect.type === "kick_arc") {
          const [originX, originY] = effect.position;
          const dir = effect.direction || [1.0, 0.0];
          const mag = Math.hypot(dir[0], dir[1]) || 1.0;
          const nx = dir[0] / mag;
          const ny = dir[1] / mag;
          const px = -ny;
          const py = nx;
          const duration = Math.max(0.001, Number(effect.duration || 0.18));
          const progress = effectProgress(effect, duration);
          const fadeIn = clamp(progress / 0.2, 0.0, 1.0);
          const fadeOut = clamp((1.0 - progress) / 0.3, 0.0, 1.0);
          const visibility = Math.min(fadeIn, fadeOut);
          const frame = Math.min(4, Math.floor(progress * 5));
          const frameReach = [14.0, 17.0, 20.0, 17.0, 13.0][frame];
          const frameSpread = [7.0, 9.5, 11.0, 8.5, 6.5][frame];
          const frameThickness = [2, 3, 3, 2, 1][frame];
          const particleCount = [8, 10, 12, 9, 6][frame];
          const baseTravel = 3.0 + progress * 16.0;

          const x = originX + nx * baseTravel;
          const y = originY + ny * baseTravel;
          const reach = frameReach;
          const spread = frameSpread;
          const baseX = x - nx * 3;
          const baseY = y - ny * 3;
          const coreX = x + nx * 4;
          const coreY = y + ny * 4;
          const tipX = x + nx * reach;
          const tipY = y + ny * reach;
          const leftX = x + nx * (reach - 2) + px * spread;
          const leftY = y + ny * (reach - 2) + py * spread;
          const rightX = x + nx * (reach - 2) - px * spread;
          const rightY = y + ny * (reach - 2) - py * spread;

          for (let ghost = 1; ghost <= 2; ghost += 1) {
            const shift = ghost * 3.4;
            const gx = x - nx * shift;
            const gy = y - ny * shift;
            const gCoreX = gx + nx * 4;
            const gCoreY = gy + ny * 4;
            const gTipX = gx + nx * (reach - ghost * 1.2);
            const gTipY = gy + ny * (reach - ghost * 1.2);
            drawPixelLine(gCoreX, gCoreY, gTipX, gTipY, 1.8, 1, "#fdba74", (0.18 + 0.1 * (2 - ghost)) * visibility);
          }

          drawPixelLine(baseX, baseY, tipX, tipY, 1.7, frameThickness, "#f97316", 0.84 * visibility);
          drawPixelLine(baseX + nx, baseY + ny, tipX, tipY, 2.4, 1, "#ffedd5", 0.9 * visibility);
          drawPixelLine(coreX, coreY, leftX, leftY, 2.2, 1, "#fb923c", 0.78 * visibility);
          drawPixelLine(coreX, coreY, rightX, rightY, 2.2, 1, "#fb923c", 0.78 * visibility);

          for (let i = 0; i < 5; i += 1) {
            const t = (i + 1) / 6;
            const cx = coreX + nx * (t * (reach - 8));
            const cy = coreY + ny * (t * (reach - 8));
            const halfWidth = (1.0 - t) * (spread - 1.0) + 1.0;
            drawPixelLine(
              cx - px * halfWidth,
              cy - py * halfWidth,
              cx + px * halfWidth,
              cy + py * halfWidth,
              2.4,
              1,
              "#fdba74",
              0.54 * visibility,
            );
          }

          const seedBase = Number(effect.id || 0) * 0.731;
          for (let i = 0; i < particleCount; i += 1) {
            const seed = seedBase + i * 1.913;
            const r1 = (Math.sin(seed * 0.913) + 1) * 0.5;
            const r2 = (Math.sin((seed + 11) * 1.331) + 1) * 0.5;
            const r3 = (Math.sin((seed + 27) * 0.677) + 1) * 0.5;
            const forwardDistance = 2 + r1 * (reach + 1);
            const side = (r2 - 0.5) * spread * (1.0 - Math.min(1.0, forwardDistance / (reach + 2)));
            const pxPos = x + nx * forwardDistance + px * side;
            const pyPos = y + ny * forwardDistance + py * side;
            const size = r3 > 0.65 ? 2 : 1;
            const color = r2 > 0.52 ? "#ffedd5" : "#fdba74";
            drawPixelDot(pxPos, pyPos, size, color, 0.78 * visibility);
          }

          drawPixelDot(coreX, coreY, 5, "#f97316", 0.45 * visibility);
          drawPixelDot(coreX, coreY, 3, "#ffedd5", 0.9 * visibility);
          drawPixelDot(tipX, tipY, 4, "#fff7ed", 0.92 * visibility);
          continue;
        }

        if (effect.type === "kick_hit") {
          const [baseX, baseY] = effect.position;
          const source = effect.source || [baseX - 8, baseY];
          const dir = effect.direction || [baseX - source[0], baseY - source[1]];
          const mag = Math.hypot(dir[0], dir[1]) || 1.0;
          const nx = dir[0] / mag;
          const ny = dir[1] / mag;
          const px = -ny;
          const py = nx;
          const duration = Math.max(0.001, Number(effect.duration || 0.24));
          const progress = effectProgress(effect, duration);
          const elapsed = progress * duration;
          const fadeIn = clamp(progress / 0.2, 0.0, 1.0);
          const fadeOut = clamp((1.0 - progress) / 0.32, 0.0, 1.0);
          const visibility = Math.min(fadeIn, fadeOut);
          const travelSpeed = Math.max(0.0, Number(effect.travel_speed || 0.0));
          const frame = Math.min(4, Math.floor(progress * 5));
          const travel = elapsed * travelSpeed;
          const x = baseX + nx * travel;
          const y = baseY + ny * travel;

          const leadLen = [34.0, 39.0, 32.0, 24.0, 18.0][frame];
          const backLen = [17.0, 18.0, 16.0, 13.0, 9.0][frame];
          const mainThickness = [8, 7, 6, 4, 3][frame];
          const crossHalf = [20.0, 21.0, 17.0, 13.0, 10.0][frame];
          const coreSize = [22, 19, 15, 11, 8][frame];
          const sparkCount = [58, 48, 40, 30, 22][frame];
          const scratchCount = [22, 18, 15, 11, 8][frame];

          const leadX = x + nx * leadLen;
          const leadY = y + ny * leadLen;
          const backX = x - nx * backLen;
          const backY = y - ny * backLen;

          drawPixelLine(backX, backY, leadX, leadY, 1.6, mainThickness, "#fb7185", 0.9 * visibility);
          drawPixelLine(backX, backY, leadX, leadY, 2.6, 3, "#ffe4e6", 0.92 * visibility);
          drawPixelLine(
            x - px * crossHalf,
            y - py * crossHalf,
            x + px * crossHalf,
            y + py * crossHalf,
            2.3,
            Math.max(2, mainThickness - 1),
            "#f97316",
            0.72 * visibility,
          );
          drawPixelDot(x, y, coreSize + 2, "#f97316", 0.45 * visibility);
          drawPixelDot(x, y, coreSize, "#fecdd3", 0.9 * visibility);
          drawPixelDot(leadX, leadY, Math.max(3, coreSize - 4), "#fff7ed", 0.9 * visibility);

          const seedBase = Number(effect.id || 0) * 0.557;
          for (let i = 0; i < sparkCount; i += 1) {
            const seed = seedBase + i * 1.271;
            const r1 = (Math.sin(seed * 0.873) + 1) * 0.5;
            const r2 = (Math.sin((seed + 13) * 1.197) + 1) * 0.5;
            const r3 = (Math.sin((seed + 31) * 1.617) + 1) * 0.5;
            const forwardDistance = 1.0 + r1 * (24 + frame * 4);
            const side = (r2 - 0.5) * (12.0 + frame * 2.4) * (1.0 - Math.min(1.0, forwardDistance / 34));
            const pxPos = x + nx * forwardDistance + px * side;
            const pyPos = y + ny * forwardDistance + py * side;
            const size = r3 > 0.94 ? 4 : r3 > 0.65 ? 2 : 1;
            const color = r2 > 0.5 ? "#fff1f2" : "#fda4af";
            drawPixelDot(pxPos, pyPos, size, color, Math.max(0.1, 0.98 - forwardDistance * 0.014) * visibility);
          }

          for (let i = 0; i < scratchCount; i += 1) {
            const t = (i + 1) / (scratchCount + 1);
            const sx = x - nx * (2 + t * (15 + frame * 2.0)) + px * ((i % 2 === 0 ? -1 : 1) * t * (5.0 + frame * 0.65));
            const sy = y - ny * (2 + t * (15 + frame * 2.0)) + py * ((i % 2 === 0 ? -1 : 1) * t * (5.0 + frame * 0.65));
            const ex = sx - nx * (4.2 + frame * 0.5);
            const ey = sy - ny * (4.2 + frame * 0.5);
            drawPixelLine(sx, sy, ex, ey, 1.4, 2, "#fb7185", 0.72 * visibility);
          }
          continue;
        }

        if (effect.type === "pickup") {
          const [x, y] = effect.position;
          ctx.save();
          ctx.globalAlpha = 0.8;
          ctx.strokeStyle = "#86efac";
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.arc(x, y, 11, 0, Math.PI * 2);
          ctx.stroke();
          ctx.restore();
          continue;
        }

        if (effect.type === "throw") {
          const [x, y] = effect.position;
          ctx.save();
          ctx.globalAlpha = 0.75;
          ctx.fillStyle = "#e2e8f0";
          for (let i = 0; i < 4; i += 1) {
            const angle = (i / 4) * Math.PI * 2 + animationTime * 0.7;
            const radius = 8.0;
            ctx.fillRect(x + Math.cos(angle) * radius - 1, y + Math.sin(angle) * radius - 1, 2, 2);
          }
          ctx.restore();
          continue;
        }

        if (effect.type === "door_open") {
          const [x, y] = effect.position;
          ctx.save();
          ctx.globalAlpha = 0.6;
          ctx.strokeStyle = "#f59e0b";
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.arc(x, y, 10.0, 0, Math.PI * 2);
          ctx.stroke();
          ctx.restore();
        }
      }
    }

    function drawPlayer(player) {
      const [x, y] = player.position;
      const [fx, fy] = player.facing;
      const angle = Math.atan2(fy, fx);
      const walkFrame = Math.floor(animationTime * 12) % 12;
      const character = String(player.character || "lemon").toLowerCase();
      const bodyImage = images[character] || images.lemon;
      const legsBaseIndex = {
        lemon: 0,
        orange: 12,
        lime: 24,
        grapefruit: 36,
      }[character] || 0;
      const legsIndex = legsBaseIndex + walkFrame;
      const legsSx = (legsIndex % 12) * LEG_FRAME_SIZE;
      const legsSy = Math.floor(legsIndex / 12) * LEG_FRAME_SIZE;

      let bodyIndex = 10;
      if (player.stun_remaining > 0.0) {
        bodyIndex = 40;
      } else if (player.weapon) {
        bodyIndex = player.shoot_cooldown > 0.01 ? 31 : 30;
      }

      const bodySx = (bodyIndex % 10) * CHARACTER_FRAME_SIZE;
      const bodySy = Math.floor(bodyIndex / 10) * CHARACTER_FRAME_SIZE;

      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(angle);

      ctx.drawImage(images.legs, legsSx, legsSy, LEG_FRAME_SIZE, LEG_FRAME_SIZE, -16, -16, LEG_FRAME_SIZE, LEG_FRAME_SIZE);
      ctx.drawImage(
        bodyImage,
        bodySx,
        bodySy,
        CHARACTER_FRAME_SIZE,
        CHARACTER_FRAME_SIZE,
        -32,
        -32,
        CHARACTER_FRAME_SIZE,
        CHARACTER_FRAME_SIZE,
      );

      if (player.weapon) {
        const gunIdx = player.weapon.type === "Revolver" ? 0 : 3;
        const sx = (gunIdx % 3) * CHARACTER_FRAME_SIZE;
        const sy = Math.floor(gunIdx / 3) * CHARACTER_FRAME_SIZE;
        ctx.drawImage(images.guns, sx, sy, CHARACTER_FRAME_SIZE, CHARACTER_FRAME_SIZE, 4, -18, 44, 44);
      }

      ctx.strokeStyle = player.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(0, 0, 13, 0, Math.PI * 2);
      ctx.stroke();

      ctx.restore();
    }

    function updateCamera(state) {
      const level = state.level;
      const alivePlayers = (state.players || []).filter((player) => player.alive);
      const cameraOverride = state && state.camera_override && typeof state.camera_override === "object"
        ? state.camera_override
        : null;

      let targetX = level.width / 2;
      let targetY = level.height / 2;

      if (alivePlayers.length > 0) {
        targetX = alivePlayers.reduce((acc, player) => acc + player.position[0], 0) / alivePlayers.length;
        targetY = alivePlayers.reduce((acc, player) => acc + player.position[1], 0) / alivePlayers.length;
      }

      let maxDist = 0;
      for (const player of alivePlayers) {
        const dx = player.position[0] - targetX;
        const dy = player.position[1] - targetY;
        maxDist = Math.max(maxDist, Math.hypot(dx, dy));
      }

      const aspect = canvas.width / canvas.height;
      const desiredWorldHeight = clamp(maxDist * 2 + 220, 180, level.height + 140);
      const desiredWorldWidth = desiredWorldHeight * aspect;

      let targetScale = Math.min(canvas.width / desiredWorldWidth, canvas.height / desiredWorldHeight);
      targetScale = Math.max(1.9, targetScale);

      let positionSmooth = 0.14;
      let scaleSmooth = 0.08;
      if (cameraOverride) {
        const overrideX = Number(cameraOverride.x);
        const overrideY = Number(cameraOverride.y);
        const overrideScale = Number(cameraOverride.scale);
        const overridePositionSmooth = Number(cameraOverride.position_smooth);
        const overrideScaleSmooth = Number(cameraOverride.scale_smooth);
        if (Number.isFinite(overrideX)) targetX = overrideX;
        if (Number.isFinite(overrideY)) targetY = overrideY;
        if (Number.isFinite(overrideScale) && overrideScale > 0.05) {
          targetScale = overrideScale;
        }
        if (Number.isFinite(overridePositionSmooth)) {
          positionSmooth = clamp(overridePositionSmooth, 0.01, 1.0);
        }
        if (Number.isFinite(overrideScaleSmooth)) {
          scaleSmooth = clamp(overrideScaleSmooth, 0.01, 1.0);
        }
      }

      const halfViewW = canvas.width / (2 * targetScale);
      const halfViewH = canvas.height / (2 * targetScale);

      if (level.width > halfViewW * 2) {
        targetX = clamp(targetX, halfViewW, level.width - halfViewW);
      } else {
        targetX = level.width / 2;
      }

      if (level.height > halfViewH * 2) {
        targetY = clamp(targetY, halfViewH, level.height - halfViewH);
      } else {
        targetY = level.height / 2;
      }

      if (!camera.initialized) {
        camera.x = targetX;
        camera.y = targetY;
        camera.scale = targetScale;
        camera.initialized = true;
        return;
      }

      camera.x += (targetX - camera.x) * positionSmooth;
      camera.y += (targetY - camera.y) * positionSmooth;
      camera.scale += (targetScale - camera.scale) * scaleSmooth;
    }

    function renderFrame() {
      animationTime += 1 / 60;
      resize();

      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#07070a";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      if (!latestState || !latestState.level) {
        return;
      }

      updateCamera(latestState);
      const snappedCameraX = Math.round(camera.x * camera.scale) / camera.scale;
      const snappedCameraY = Math.round(camera.y * camera.scale) / camera.scale;
      const translateX = Math.round(canvas.width / 2 - snappedCameraX * camera.scale);
      const translateY = Math.round(canvas.height / 2 - snappedCameraY * camera.scale);

      ctx.setTransform(
        camera.scale,
        0,
        0,
        camera.scale,
        translateX,
        translateY,
      );

      drawBackdrop(latestState.level);
      drawTiles(latestState.level);
      drawWorldObjects(latestState);
      drawDebris(latestState);

      for (const pickup of latestState.pickups || []) drawPickup(pickup);
      for (const projectile of latestState.projectiles || []) drawProjectile(projectile);
      drawEffects(latestState.effects || [], "exclude_blood");

      for (const player of latestState.players || []) {
        if (player.alive) drawPlayer(player);
      }

      drawEffects(latestState.effects || [], "only_blood");
    }

    function tick() {
      renderFrame();
      rafId = global.requestAnimationFrame(tick);
    }

    function start() {
      if (rafId) return;
      rafId = global.requestAnimationFrame(tick);
    }

    function stop() {
      if (!rafId) return;
      global.cancelAnimationFrame(rafId);
      rafId = 0;
    }

    function setState(state) {
      latestState = state;
      if (state) {
        syncEffectTimeline(state);
      }
    }

    function resetCamera() {
      camera.initialized = false;
    }

    function destroy() {
      stop();
      latestState = null;
      effectTimeline.clear();
      doorMotion.clear();
      backgroundSceneCache.clear();
      parallax.initialized = false;
      parallax.x = 0;
      parallax.y = 0;
      lastStateTick = -1;
      resetCamera();
    }

    return {
      canvas,
      ctx,
      loadAssets,
      resize,
      start,
      stop,
      destroy,
      resetCamera,
      renderFrame,
      setState,
      getState() {
        return latestState;
      },
    };
  }

  class GameRenderer {
    constructor(options) {
      this._renderer = createRenderer(options || {});
      this.canvas = this._renderer.canvas;
      this.ctx = this._renderer.ctx;
    }

    loadAssets() {
      return this._renderer.loadAssets();
    }

    resize() {
      return this._renderer.resize();
    }

    start() {
      return this._renderer.start();
    }

    stop() {
      return this._renderer.stop();
    }

    destroy() {
      return this._renderer.destroy();
    }

    resetCamera() {
      return this._renderer.resetCamera();
    }

    renderFrame() {
      return this._renderer.renderFrame();
    }

    setState(state) {
      return this._renderer.setState(state);
    }

    getState() {
      return this._renderer.getState();
    }
  }

  global.GaicaGameRenderer = {
    GameRenderer,
    createDefaultAssetPaths,
  };
})(window);
