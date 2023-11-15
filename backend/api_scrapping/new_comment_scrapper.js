
// const puppeteer = require('puppeteer');
// const fs = require('fs');

// class CommentScraper {
//     constructor(page) {
//         this.page = page
//         this.todosLosComentarios = [];
//         this.numScrollInVideo = 15
//     }

//     async cerrarNavegador() {
//         await this.page.close();
//     }


//     async scrollDown() {
//         await this.page.evaluate(async () => {
//             window.scrollBy(0, window.innerHeight);
           
//         });
//     }


//     async abrir_sub_comentario() {
//         await this.page.evaluate(() => {
//             const bnt_expander_comments = document.querySelectorAll("#expander .more-button ");
//             // verifico si hay respuesta en ese comentario
//             if (bnt_expander_comments.length > 0) {
//                 // Iterar sobre cada elemento y hacer click
//                 for (const btn of bnt_expander_comments) {
//                     btn.click();
//                 }
//             } else {
//                 console.log("No se encontraron elementos con el selector proporcionado.");
//             }
//         })
//     }
    

//     async recopilarComentarios(videos) {
//         for (let i = 0; i < videos.length; i++) {
//             await this.page.goto(videos[i]);
//             const video = videos[i];
//             //cierro modal de cookies
//             this.cerrarModal(this.page)
            
//             for (let j = 0; j < this.numScrollInVideo; j++) {
//                 //scroleo
//                 await this.scrollDown();
//                 // abro los subcomentarios
//                 this.abrir_sub_comentario()
//                 // espero 1 segundo
//                 await new Promise(resolve => setTimeout(resolve, 1000));
//             }
            
//             try{
//                 await this.page.waitForSelector('#comments');

//                 const comentarios = await this.page.evaluate(() => {
//                     const comentarios = [];
//                     const content = document.querySelector('#comments #contents');
//                     const contentLength = content.children.length;
//                     const currentUrl = window.location.href;

//                     for (let k = 0; k < contentLength; k++) {
//                         const contentElement = content.children[k];
//                         const img_autor = contentElement.querySelector('#author-thumbnail #img')?.src || 'vacio';
//                         const link_autor = contentElement.querySelector('#header-author h3')?.childNodes[1]?.href || 'vacio';
//                         const autorComment = contentElement.querySelector('#author-text')?.innerText || 'vacio';
//                         const textComment = contentElement.querySelector('#content-text')?.innerText || 'vacio';
//                         const link_video = currentUrl

//                         let comentario = {
//                             'img_autor':img_autor,
//                             'link_autor':link_autor,
//                             'autor': autorComment,
//                             'comentario': textComment,
//                             'link_video' : link_video
//                         };
                        
//                         comentarios.push(comentario);

//                         const replies = contentElement.querySelectorAll("#replies #contents")
//                         // pregunto si el comenterio tiene respuestas
//                         if (replies.length > 0){
//                             for(let l = 0; l < replies.length; l++){
//                                 const elemento = replies[l]
//                                 const img_autor = elemento.querySelector('#author-thumbnail #img')?.src || 'vacio';
//                                 const link_autor = elemento.querySelector('#header-author h3')?.childNodes[1]?.href || 'vacio';
//                                 const autorComment = elemento.querySelector('#author-text')?.innerText || 'vacio';
//                                 const textComment = elemento.querySelector('#content-text')?.innerText || 'vacio';
//                                 const link_video = currentUrl

//                                 let comentario = {
//                                     'img_autor':img_autor,
//                                     'link_autor':link_autor,
//                                     'autor': autorComment,
//                                     'comentario': textComment,
//                                     'link_video' : link_video
//                                 };
                                
//                                 comentarios.push(comentario);
//                             }
//                         }    
                     
//                     }


//                     return comentarios;
//                 });

//                 this.todosLosComentarios.push({ 'link-video': video, 'comentarios': comentarios });
//                 // this.mostrarComentarios();
//                 console.log('Se han analizado ' + this.todosLosComentarios.length + ' videos');
//                 await new Promise(resolve => setTimeout(resolve, 1000));
//             } catch (error)  {
//                 console.log(`No se pudo encontrar el elemento #comments en el video ${video}. Continuando con el siguiente video.`);
//             }
                   
                
//         }
//         return this.todosLosComentarios
//     }


//     async monitorizarNuevosComentarios(videoUrl) {
//         await this.page.goto(videoUrl);
//         await this.cerrarModal(this.page);
//         await this.page.waitForSelector('#comments');

//         const observador = new MutationObserver(async () => {
//             console.log('¡Nuevos comentarios detectados!');
//             await this.recopilarNuevosComentarios(videoUrl);
//         });

//         observador.observe(document.querySelector('#comments'), {
//             childList: true,
//             subtree: true,
//         });
//     }



































    
    
//     mostrarComentarios() {
//         for (let elemento of this.todosLosComentarios) {
//             console.log(`Comentarios para el video: ${elemento['link-video']}`);
//             for (let comentario of elemento['comentarios']) {
//                 console.log(comentario);
//             }
//         }
//         console.log('Se han analizado ' + this.todosLosComentarios.length + ' videos');
//         return this.todosLosComentarios
//     }


//     async cerrarModal(page) {
//         const modalSelector = '#dialog';
//         try {
//             await page.waitForSelector(modalSelector, { visible: true, timeout: 5000 }); //iempo de espera máximo
//             const modalElement = await page.$(modalSelector);
//             if (modalElement) {
//                 const closeButtonSelector = '.yt-spec-button-shape-next--filled';
//                 const closeButton = await modalElement.$(closeButtonSelector);
//                 if (closeButton) {
//                     await closeButton.click();
//                 }
//             }
//         } catch (error) {
//             // console.log('No se encontró el modal o no era visible en esta página.');
//         }
//     }
// }

// module.exports = CommentScraper;
