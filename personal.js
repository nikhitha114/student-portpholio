document.addEventListener("DOMContentLoaded",()=>{
    document.querySelectorAll('.portfolio-box').forEach(box=>{
        box.addEventListener('click',()=>{
            box.classList.toggle('active');
            document.querySelectorAll('.portfolio-box').forEach(otherBox => {
                if(otherBox !== box){
                    otherBox.classList.remove('active');
                }
            });
        });
    });
    const form = document.getElementById('contactForm');
    form.addEventListener('submit', (event) => {
        const email = form.email.value;
        const e_pattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!e_pattern.test(email)) {
            alert('Please enter a valid email address.');
            event.preventDefault();
        }
    });
    const navLinks = document.querySelectorAll('nav ul li a');
    const sections = document.querySelectorAll('section');
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.3
    };
    const observer=new IntersectionObserver((entries,observer)=>{
        entries.forEach(entry=>{
            if(entry.isIntersecting){
                navLinks.forEach(link=>{
                    link.classList.toggle('active',link.getAttribute('href').substring(1)===entry.target.id);
                });
            }
        });
    },observerOptions);
    sections.forEach(section=>{
        observer.observe(section);
    });
});