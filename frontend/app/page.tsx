import Link from "next/link";

export default function Home() {
  return (
    <main>
      <section className="card">
        <h2>What this is</h2>
        <p>
          Tayr detects, tracks and classifies small aerial objects in recorded video, and
          measures how well it does so. It is a research and analysis tool: it performs
          detection, tracking and classification only.
        </p>
        <p className="muted">
          Upload a video and it is decoded in an isolated worker, tracked, and reported on.
          Nothing is analysed in real time and no footage is used for anything else.
        </p>
      </section>

      <section className="card">
        <h2>Current status</h2>
        <p className="notice">
          No trained detector exists yet. Jobs run the full pipeline — probe, decode, track,
          extract motion features — but with a placeholder detector that finds nothing. Every
          such result is labelled synthetic, in the interface and in the data.
        </p>
      </section>

      <div className="row">
        <Link href="/login"><button>Sign in</button></Link>
        <Link href="/register"><button className="secondary">Create an account</button></Link>
      </div>
    </main>
  );
}
