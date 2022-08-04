package day12;
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;

public class LambdaGui extends JFrame{

	JPanel p=new JPanel();
	JPanel pN=new JPanel();
	JLabel lb;
	JButton bt1, bt2;
	public LambdaGui() {
		super("::LambdaGui::");
		
		add(p,"Center");
		p.setLayout(new BorderLayout());
		
		lb=new JLabel(new ImageIcon("myicon.PNG"));
		lb.setText("Welcome to MyApp");
		lb.setHorizontalTextPosition(JLabel.CENTER);
		lb.setVerticalTextPosition(JLabel.TOP);
		lb.setFont(new Font("sans-serif",Font.BOLD,24));
							//서체		스타일		크기
		
		p.add(pN,"North");
		p.add(lb,"Center");
		
		bt1=new JButton("Blue");
		bt2=new JButton("Pink");
		pN.add(bt1);
		pN.add(bt2);
		
		
		//[1] bt1 클릭시 lb의 글자색을 파랑으로 =>Anonymous class
		bt1.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				lb.setForeground(Color.blue);
			}
		});
		
		//[2] bt2 핑크색 => Lambda식 이용해서 이벤트 처리하기
		bt2.addActionListener(e->{lb.setForeground(Color.pink);});
		
		this.setSize(500,700);
		this.setVisible(true);
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

	public static void main(String[] args) {
		
		new LambdaGui();
	}

}
