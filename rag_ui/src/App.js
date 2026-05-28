import logo from './logo.svg';
import './App.css';
import { useState } from 'react';

function App() {
  const [question, setQuestion] = useState("")
  const [response, setResponse] = useState("")
  const [loading, setLoading] = useState(false)

  const askQuestion = async () => {
    if (!question.trim())
      return

    setLoading(true)
    setResponse("")
    try{
      const result = await fetch("http://localhost:8000/ask", {
        method: "post",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({question})
      })
      const data = await result.json()
      console.log(data)
      setResponse(data.answer)

    } catch (error){
      console.assert(error)
    }
    finally{
      setLoading(false)
    }
  }

  return (
    <div className="App">
      <div style={{width: "600px", margin:"50px auto", fontFamily:"Arial"}}>
        <h2> ACER Question Answer UI </h2>
        <textarea rows="4" value={question} onChange={(e)=> setQuestion(e.target.value)}
          placeholder='Enter your question here....'>
          style={{
            width: "100%", padding: "10px", margin: "10px"
          }}
        </textarea>
        <br/>
        <button onClick={askQuestion} 
          style={{padding: "10px 20px", cursor: "pointer"}}>
            {loading?"Loading...." : "Submit"}
        </button>
        <br/>
        {
          response && (
            <div style={{
              marginTop: "20px", 
              padding: "15px",
              border: "1px solid #ccc",
              borderRadius: "5px"
            }}>
              <strong>
                response: <p>{response}</p>
              </strong>
            </div>
          )
        }
      </div>
    </div>
  );
}

export default App;
