export function socketController(req , res){
    const { trueOrFalse} = req.body();
    res.json({
        data : "you can connect socket" 
    })
}