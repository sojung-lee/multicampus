package day11;
import javax.swing.*;
import java.awt.*;

public class MyApp_정은 extends JFrame{
	
	JButton bt1, bt2, bt3, bt4;
	JPanel p = new JPanel();
	JPanel pN = new JPanel();
	JTextField tf;
	JTextArea ta;

	public MyApp_정은() {
		super("::MyApp::");
		add(p);
		add(pN,"North");
		p.setLayout(new BorderLayout());
		p.setBackground(Color.white);
		pN.setBackground(Color.pink);
		
		tf = new JTextField(20);
		tf.setBackground(Color.cyan);
		
		ta = new JTextArea("JTextArea 입니다", 5, 15);
		
		bt1 = new JButton("Login");
		bt2 = new JButton("Join");
		bt3 = new JButton("Home");
		bt4 = new JButton("Exit");
		
		pN.add(bt1);
		pN.add(bt2);
		pN.add(bt3);
		pN.add(bt4);
		
		p.add(tf);
		
		JScrollPane sp = new JScrollPane(ta);
		p.add(sp);
		sp.setForeground(Color.green);//안뜹니다
		
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

	public static void main(String[] args) {
		MyApp_정은 my = new MyApp_정은();
		my.setSize(500, 500);
		my.setVisible(true);

	}

}
