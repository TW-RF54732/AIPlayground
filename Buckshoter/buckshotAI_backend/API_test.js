myData = {"customHealth":2,"custombullet":null,"customItem":[["beer"],["beer"]]}

fetch("api/game/initGame",{
        method:"POST",
        headers : { "Content-Type": "application/json" },
        body : JSON.stringify(myData)
    }
)
.then(response => console.log(response))
.catch(error => console.log(error));

fetch("api/game/getStatus").then(async response =>{
    const data = await response.json()
    console.log(data)
})