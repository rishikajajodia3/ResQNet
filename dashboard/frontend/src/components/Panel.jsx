export function Panel({ title, hint, tag, right, children, flush, hero, className }) {
  return (
    <section className={`panel${hero ? " hero" : ""}${className ? " " + className : ""}`}>
      {(title || tag || hint || right) && (
        <header>
          {title && <h3>{title}</h3>}
          {tag}
          {hint && !right && <span className="hint">{hint}</span>}
          {right && <div className="head-right">{right}</div>}
        </header>
      )}
      <div className={flush ? "body flush" : "body"}>{children}</div>
    </section>
  );
}

export function Loading({ label = "Loading" }) {
  return (
    <div className="state-msg">
      <span className="spinner" />
      {label}…
    </div>
  );
}

export function ErrorState({ error, hint }) {
  return (
    <div className="state-msg error">
      Could not load data.
      <span className="mono">{String(error?.message || error)}</span>
      {hint && <span className="mono">{hint}</span>}
    </div>
  );
}

export function EmptyState({ label, action }) {
  return (
    <div className="state-msg">
      {label}
      {action && <span className="mono">{action}</span>}
    </div>
  );
}
