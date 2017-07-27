/**
 * Created by Yair Hadas on 7/24/2017.
 */

//document.getElementById("demo").innerHTML = "My First JavaScript";


//import io from 'socket.io-client';

var socket = io.connect('http://localhost');
socket.on('news', function (data) {
    console.log(data);
    socket.emit('my other event', { my: 'data' });
     });

