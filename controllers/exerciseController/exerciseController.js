import prisma from "../../db/prisma.js"

export async function exerciseController(req , res){
    const exercises = await prisma.exercise.findFirst();
    res.json({data : exercises.stepImages});
}

/* 
1. yashyas : first page intro and solution
2. client team :
3. vivek 
4. soham :
5. go to demonstration : yashyas yt vedio explain ok
6. webcam demonstration : if done and time available in meet


solution : 
problem : 
    virtual exercise "assistant"
points : 
    no diagonsis , real time vedio demonstration , stpes to perform the exercise,

tech :
    ml model , socket , servers ...



probable question :
1. patient input data : 
    injury type -> available exercise -> show 
    real time vedio demonstration core part ... yada yada yada
    

*/