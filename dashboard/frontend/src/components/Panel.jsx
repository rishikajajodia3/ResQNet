export function Panel({ title, hint, tag, children, flush }) {
  return (
    <section className="panel">
      <header>
        <h3>{title}</h3>
        {tag}
        {hint && <span className="hint">{hint}</span>}
      </header>
      <div className={flush ? "body flush" : "body"}>{children}</div>
    </section>
  );
}

export function Loading({ label = "Loading" }) {
  return (
    <div className="state-msg">
      <span className="spinner" />
      {label}...
    </div>
  );
}

export function ErrorState({ error }) {
  return (
    <div className="state-msg error">
      Could not load data.
      <div className="mono" style={{ marginTop: 6, fontSize: 12 }}>
        {String(error.message || error)}
      </div>
    </div>
  );
}

export function EmptyState({ label }) {
  return <div className="state-msg">{label}</div>;
}
