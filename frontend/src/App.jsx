import { useState } from 'react'
import './App.css'

function App() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleSubmit(event) {
    event.preventDefault()

    const trimmedQuery = query.trim()
    if (!trimmedQuery) {
      setResults([])
      setError('Enter a search query first.')
      return
    }

    setLoading(true)
    setError('')

    try {
      const res = await fetch(`${import.meta.env.VITE_API_URL ?? 'http://localhost:8000'}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: trimmedQuery, top_k: 5 }),
      })

      if (!res.ok) {
        throw new Error(`Request failed with status ${res.status}`)
      }

      const data = await res.json()
      setResults(data.results ?? [])
    } catch (err) {
      setResults([])
      setError(err instanceof Error ? err.message : 'Search failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="app-shell">
      <section className="search-card">
        <p className="eyebrow">arXiv semantic search</p>
        <h1>Search machine learning papers by meaning.</h1>
        <p className="lede">
          Type a concept, send it to the FastAPI backend, and inspect the most
          relevant arXiv abstracts.
        </p>

        <form className="search-form" onSubmit={handleSubmit}>
          <label className="sr-only" htmlFor="query">
            Search query
          </label>
          <input
            id="query"
            type="text"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Try 'graph neural networks'"
          />
          <button type="submit" disabled={loading}>
            {loading ? 'Searching...' : 'Search'}
          </button>
        </form>

        {error ? <p className="error">{error}</p> : null}
      </section>

      <section className="results-panel">
        <div className="results-header">
          <h2>Results</h2>
          <p>{results.length} papers</p>
        </div>

        <div className="results-list">
          {results.map((result, index) => (
            <article className="result-card" key={`${result.title}-${index}`}>
              <div className="result-meta">
                <span>#{index + 1}</span>
                <span>score {result.score.toFixed(3)}</span>
              </div>
              <h3>{result.title}</h3>
              <p>{result.abstract}</p>
            </article>
          ))}
        </div>
      </section>
    </main>
  )
}

export default App
